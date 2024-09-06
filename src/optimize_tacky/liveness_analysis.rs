use std::collections::{HashMap, HashSet};

use ecow::EcoString;

use crate::{
    control_flow::{Cfg, Node, NodeId},
    semantics::type_check::{Attr, SymbolTable},
    tacky::{Instruction, Val},
};

pub fn eliminate_dead_stores(graph: &mut Cfg<Instruction>, symbol_table: &SymbolTable) {
    let mut annotation = Annotation {
        block_annotation: HashMap::new(),
        instruction_annotation: HashMap::new(),
    };

    let aliased_vals = graph
        .all_instructions()
        .filter_map(|inst| {
            if let Instruction::GetAddress { src, .. } = inst {
                Some(src.clone())
            } else {
                None
            }
        })
        .collect();

    annotation.iterate(graph, symbol_table, &aliased_vals);
    annotation.rewrite_instructions(graph);
}

#[derive(Debug)]
struct Annotation {
    block_annotation: HashMap<usize, HashSet<EcoString>>,
    instruction_annotation: HashMap<usize, Vec<HashSet<EcoString>>>,
}

impl Annotation {
    fn init_block(&mut self, block: &Node<Instruction>) {
        self.block_annotation.insert(block.id, HashSet::new());
        self.instruction_annotation
            .insert(block.id, vec![HashSet::new(); block.instructions.len()]);
    }

    fn annotate_block(&mut self, block_id: usize, live_variables: HashSet<EcoString>) {
        self.block_annotation.insert(block_id, live_variables);
    }

    fn annotate_instruction(
        &mut self,
        block_id: usize,
        inst_index: usize,
        live_variables: HashSet<EcoString>,
    ) {
        self.instruction_annotation.get_mut(&block_id).unwrap()[inst_index] = live_variables;
    }

    fn transfer(
        &mut self,
        block: &Node<Instruction>,
        end_live_variables: &HashSet<EcoString>,
        all_static_vars: &HashSet<EcoString>,
        aliased_vals: &HashSet<Val>,
    ) {
        let mut current_live_variables = end_live_variables.clone();
        let all_aliased_vars = aliased_vals
            .iter()
            .map(|v| v.var())
            .cloned()
            .collect::<HashSet<_>>();

        for (i, inst) in block.instructions.iter().enumerate().rev() {
            self.annotate_instruction(block.id, i, current_live_variables.clone());

            match inst {
                Instruction::Binary {
                    lhs: src1,
                    rhs: src2,
                    dst,
                    ..
                }
                | Instruction::AddPtr {
                    ptr: src1,
                    index: src2,
                    dst,
                    ..
                } => {
                    remove(&mut current_live_variables, dst);
                    insert(&mut current_live_variables, src1);
                    insert(&mut current_live_variables, src2);
                }
                Instruction::Unary { src, dst, .. }
                | Instruction::Cast { src, dst }
                | Instruction::Copy { src, dst } => {
                    remove(&mut current_live_variables, dst);
                    insert(&mut current_live_variables, src);
                }
                Instruction::Load { src, dst } | Instruction::Store { src, dst } => {
                    insert(&mut current_live_variables, src);
                    insert(&mut current_live_variables, dst);
                    // current_live_variables.extend(all_static_vars.iter().cloned());
                    current_live_variables.extend(all_aliased_vars.iter().cloned());
                }
                Instruction::JumpIfNotZero { src, .. } | Instruction::JumpIfZero { src, .. } => {
                    insert(&mut current_live_variables, src);
                }
                Instruction::FunCall { args, dst, .. } => {
                    if let Some(dst) = dst {
                        remove(&mut current_live_variables, dst);
                    }
                    for arg in args {
                        insert(&mut current_live_variables, arg);
                    }

                    current_live_variables.extend(all_static_vars.iter().cloned());
                    current_live_variables.extend(all_aliased_vars.iter().cloned());
                }
                Instruction::Return(Some(dst)) | Instruction::GetAddress { dst, .. } => {
                    insert(&mut current_live_variables, dst);
                }
                Instruction::CopyFromOffset { src, .. } => {
                    // remove(&mut current_live_variables, dst);
                    insert(&mut current_live_variables, src);
                }
                Instruction::CopyToOffset { src, .. } => {
                    // remove(&mut current_live_variables, &Val::Var(dst.clone()));
                    insert(&mut current_live_variables, src);
                }
                Instruction::Nop
                | Instruction::Jump(_)
                | Instruction::Label(_)
                | Instruction::Return(None) => {}
            }
        }

        self.annotate_block(block.id, current_live_variables);
    }

    fn meet(
        &mut self,
        block: &Node<Instruction>,
        all_static_variables: &HashSet<EcoString>,
    ) -> HashSet<EcoString> {
        let mut live_variables = HashSet::new();

        for succ in &block.successors {
            match succ {
                NodeId::Exit => {
                    live_variables.extend(all_static_variables.iter().cloned());
                }
                NodeId::Entry => unreachable!(),
                NodeId::Block(id) => {
                    let succ_live_variables = self.block_annotation.get(id).unwrap();
                    live_variables.extend(succ_live_variables.iter().cloned());
                }
            }
        }

        live_variables
    }

    fn iterate(
        &mut self,
        graph: &Cfg<Instruction>,
        symbol_table: &SymbolTable,
        aliased_vals: &HashSet<Val>,
    ) {
        let all_static_vars: HashSet<EcoString> = symbol_table
            .iter()
            .filter_map(|(k, v)| {
                if matches!(v, Attr::Static { .. }) {
                    Some(k.clone())
                } else {
                    None
                }
            })
            .collect();

        let mut worklist = Vec::new();
        for node in graph.nodes.values() {
            self.init_block(node);
            worklist.push(node);
        }

        while let Some(block) = worklist.pop() {
            let old_annotations = self.block_annotation[&block.id].clone();
            let incoming = self.meet(block, &all_static_vars);
            self.transfer(block, &incoming, &all_static_vars, aliased_vals);

            if old_annotations != self.block_annotation[&block.id] {
                for pred in &block.predecessors {
                    if let NodeId::Block(id) = pred {
                        let pred_node = &graph.nodes[id];
                        if worklist.iter().all(|n| n.id != pred_node.id) {
                            worklist.push(pred_node);
                        }
                    }
                }
            }
        }
    }

    fn rewrite_instructions(&self, graph: &mut Cfg<Instruction>) {
        for node in graph.nodes.values_mut() {
            for (i, inst) in node.instructions.iter_mut().enumerate() {
                let live_variables = &self.instruction_annotation[&node.id][i];
                if let Some(dst) = dst_field(inst) {
                    if !live_variables.contains(dst) {
                        *inst = Instruction::Nop;
                    }
                }
            }
        }
    }
}

fn dst_field(inst: &Instruction) -> Option<&EcoString> {
    match inst {
        Instruction::Binary { dst, .. }
        | Instruction::Unary { dst, .. }
        | Instruction::AddPtr { dst, .. }
        | Instruction::Load { dst, .. }
        | Instruction::Cast { dst, .. }
        | Instruction::Copy { dst, .. }
        | Instruction::GetAddress { dst, .. }
        | Instruction::CopyFromOffset { dst, .. }
        | Instruction::CopyToOffset { dst, .. } => Some(dst.var()),
        _ => None,
    }
}

fn remove(set: &mut HashSet<EcoString>, val: &Val) -> bool {
    if let Val::Var(var) = val {
        set.remove(var)
    } else {
        false
    }
}

fn insert(set: &mut HashSet<EcoString>, val: &Val) -> bool {
    if let Val::Var(var) = val {
        set.insert(var.clone())
    } else {
        false
    }
}
