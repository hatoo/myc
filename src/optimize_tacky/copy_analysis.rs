use std::collections::{HashMap, HashSet};

use crate::{
    semantics::type_check::SymbolTable,
    tacky::{Instruction, Val},
};

use super::graph::{Graph, Node, NodeId};

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub struct Copy {
    pub src: Val,
    pub dst: Val,
}

#[derive(Debug)]
struct Annotation {
    block_annotation: HashMap<usize, HashSet<Copy>>,
    instruction_annotation: HashMap<usize, Vec<HashSet<Copy>>>,
}

pub fn copy_propagation(graph: &mut Graph, symbol_table: &SymbolTable) {
    let mut annotation = Annotation {
        block_annotation: HashMap::new(),
        instruction_annotation: HashMap::new(),
    };

    annotation.iterate(graph, symbol_table);
    annotation.rewrite_instructions(graph);
}

impl Annotation {
    fn init_block(&mut self, block: &Node) {
        self.instruction_annotation
            .insert(block.id, vec![HashSet::new(); block.instructions.len()]);
    }

    fn annotate_instruction(
        &mut self,
        block_id: usize,
        inst_index: usize,
        reaching_copies: HashSet<Copy>,
    ) {
        self.instruction_annotation.get_mut(&block_id).unwrap()[inst_index] = reaching_copies;
    }

    fn transfer(
        &mut self,
        block: &Node,
        symbol_table: &SymbolTable,
        initial_reaching_copies: &HashSet<Copy>,
        aliased_vals: &HashSet<Val>,
    ) {
        let mut current_reaching_copies = initial_reaching_copies.clone();
        for (i, inst) in block.instructions.iter().enumerate() {
            self.annotate_instruction(block.id, i, current_reaching_copies.clone());
            match inst {
                Instruction::Copy { dst, src } => {
                    let cpy = Copy {
                        src: src.clone(),
                        dst: dst.clone(),
                    };
                    if !current_reaching_copies.contains(&cpy) {
                        current_reaching_copies.retain(|c| !(c.src == *dst || c.dst == *dst));
                        current_reaching_copies.insert(cpy);
                    }
                }
                Instruction::CopyFromOffset { src, dst, .. } => {
                    let cpy = Copy {
                        src: Val::Var(src.clone()),
                        dst: dst.clone(),
                    };
                    if !current_reaching_copies.contains(&cpy) {
                        current_reaching_copies.retain(|c| !(c.src == *dst || c.dst == *dst));
                    }
                }
                Instruction::CopyToOffset { src, dst, .. } => {
                    let cpy = Copy {
                        src: src.clone(),
                        dst: Val::Var(dst.clone()),
                    };
                    if !current_reaching_copies.contains(&cpy) {
                        current_reaching_copies.retain(|c| !(c.src == cpy.dst || c.dst == cpy.dst));
                    }
                }
                Instruction::FunCall { dst, .. } => {
                    current_reaching_copies.retain(|c| {
                        !(c.src.is_static(symbol_table)
                            || c.dst.is_static(symbol_table)
                            || Some(&c.src) == dst.as_ref()
                            || Some(&c.dst) == dst.as_ref()
                            || aliased_vals.contains(&c.src)
                            || aliased_vals.contains(&c.dst))
                    });
                }
                Instruction::Unary { dst, .. }
                | Instruction::Binary { dst, .. }
                | Instruction::Cast { dst, .. } => {
                    current_reaching_copies.retain(|c| !(&c.src == dst || &c.dst == dst));
                }
                Instruction::Store { .. } => {
                    current_reaching_copies.retain(|c| {
                        !(aliased_vals.contains(&c.src)
                            || aliased_vals.contains(&c.dst)
                            || c.src.is_static(symbol_table)
                            || c.dst.is_static(symbol_table))
                    });
                }
                _ => {}
            }
        }

        self.block_annotation
            .insert(block.id, current_reaching_copies.clone());
    }

    fn meet(&mut self, block: &Node, all_copies: &HashSet<Copy>) -> HashSet<Copy> {
        let mut incoming_copies = all_copies.clone();

        for pred in &block.predecessors {
            if let NodeId::Block(id) = pred {
                let pred_copies = self.block_annotation.get(id).unwrap();
                incoming_copies = incoming_copies.intersection(pred_copies).cloned().collect();
            } else {
                return HashSet::new();
            }
        }

        incoming_copies
    }

    fn iterate(&mut self, graph: &Graph, symbol_table: &SymbolTable) {
        let all_copies = graph.all_copy_instructions();
        let aliased_vals = graph.aliased_vals();

        let mut worklist = Vec::new();
        for node in graph.nodes.values() {
            self.init_block(node);
            self.block_annotation.insert(node.id, all_copies.clone());
            worklist.push(node);
        }

        while let Some(block) = worklist.pop() {
            let old_annotations = self.block_annotation[&block.id].clone();
            let incoming_copies = self.meet(block, &all_copies);
            self.transfer(block, symbol_table, &incoming_copies, &aliased_vals);

            if old_annotations != self.block_annotation[&block.id] {
                for succ in &block.successors {
                    if let NodeId::Block(id) = succ {
                        let succ_node = &graph.nodes[id];
                        if worklist.iter().all(|n| n.id != succ_node.id) {
                            worklist.push(succ_node);
                        }
                    }
                }
            }
        }
    }

    fn rewrite_instructions(&self, graph: &mut Graph) {
        for node in graph.nodes.values_mut() {
            for (inst, anno) in node
                .instructions
                .iter_mut()
                .zip(self.instruction_annotation[&node.id].iter())
            {
                match inst {
                    Instruction::Copy { src, dst } => {
                        if anno.iter().any(|c| {
                            (&c.src == src && &c.dst == dst) || (&c.src == dst && &c.dst == src)
                        }) {
                            *inst = Instruction::Nop;
                        } else {
                            *src = replace_operand(src.clone(), anno);
                        }
                    }
                    Instruction::Unary { src, .. }
                    | Instruction::Cast { src, .. }
                    | Instruction::Load { src, .. }
                    | Instruction::Store { src, .. }
                    | Instruction::CopyToOffset { src, .. }
                    | Instruction::JumpIfNotZero { src, .. }
                    | Instruction::JumpIfZero { src, .. }
                    | Instruction::Return(Some(src)) => {
                        *src = replace_operand(src.clone(), anno);
                    }
                    Instruction::CopyFromOffset { src, .. } => {
                        if let Val::Var(var) = replace_operand(Val::Var(src.clone()), anno) {
                            *src = var
                        }
                    }
                    Instruction::Binary {
                        lhs: src1,
                        rhs: src2,
                        ..
                    }
                    | Instruction::AddPtr {
                        ptr: src1,
                        index: src2,
                        ..
                    } => {
                        *src1 = replace_operand(src1.clone(), anno);
                        *src2 = replace_operand(src2.clone(), anno);
                    }
                    Instruction::FunCall { callee, args, .. } => {
                        *callee = replace_operand(callee.clone(), anno);
                        for arg in args {
                            *arg = replace_operand(arg.clone(), anno);
                        }
                    }

                    _ => {}
                }
            }
        }
    }
}

fn replace_operand(val: Val, reaching_copies: &HashSet<Copy>) -> Val {
    match val {
        Val::Constant(_) => val,
        Val::Var(_) => {
            for c in reaching_copies {
                if c.dst == val {
                    return c.src.clone();
                }
            }
            val
        }
    }
}
