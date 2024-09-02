use std::collections::{HashMap, HashSet};

use ecow::EcoString;

use crate::{
    codegen::{Instruction, Operand, Register},
    control_flow::{self, Cfg},
    semantics::type_check::{Attr, SymbolTable},
};

use super::{is_scalar, NodeId};

#[derive(Debug)]
struct Annotation {
    block_annotation: HashMap<usize, HashSet<NodeId>>,
    instruction_annotation: HashMap<usize, Vec<HashSet<NodeId>>>,
}

impl Annotation {
    fn init_block(&mut self, block: &control_flow::Node<Instruction>) {
        self.block_annotation.insert(block.id, HashSet::new());
        self.instruction_annotation
            .insert(block.id, vec![HashSet::new(); block.instructions.len()]);
    }

    fn annotate_block(&mut self, block_id: usize, live_variables: HashSet<NodeId>) {
        self.block_annotation.insert(block_id, live_variables);
    }

    fn annotate_instruction(
        &mut self,
        block_id: usize,
        inst_index: usize,
        live_variables: HashSet<NodeId>,
    ) {
        self.instruction_annotation.get_mut(&block_id).unwrap()[inst_index] = live_variables;
    }

    fn transfer(
        &mut self,
        block: &control_flow::Node<Instruction>,
        symbol_table: &SymbolTable,
        end_live_variables: &HashSet<NodeId>,
    ) {
        let mut current_live_variables = end_live_variables.clone();

        for (i, inst) in block.instructions.iter().enumerate().rev() {
            self.annotate_instruction(block.id, i, current_live_variables.clone());

            match inst {
                _ => todo!(),
            }
        }

        self.annotate_block(block.id, current_live_variables);
    }

    fn meet(&mut self, block: &control_flow::Node<Instruction>) -> HashSet<NodeId> {
        let mut live_variables = HashSet::new();

        for succ in &block.successors {
            match succ {
                _ => todo!(),
            }
        }

        live_variables
    }

    fn iterate(&mut self, graph: &Cfg<Instruction>, symbol_table: &SymbolTable) {
        let mut worklist = Vec::new();
        for node in graph.nodes.values() {
            self.init_block(node);
            worklist.push(node);
        }

        while let Some(block) = worklist.pop() {
            let old_annotations = self.block_annotation[&block.id].clone();
            let incoming = self.meet(block);
            self.transfer(block, symbol_table, &incoming);

            if old_annotations != self.block_annotation[&block.id] {
                for pred in &block.predecessors {
                    if let control_flow::NodeId::Block(id) = pred {
                        let pred_node = &graph.nodes[id];
                        if worklist.iter().all(|n| n.id != pred_node.id) {
                            worklist.push(pred_node);
                        }
                    }
                }
            }
        }
    }
}

fn find_used_and_updated(inst: &Instruction) -> (Vec<&Operand>, Vec<&Operand>) {
    match inst {
        Instruction::Mov { src, dst, .. }
        | Instruction::MovZeroExtend { src, dst, .. }
        | Instruction::Movsx { src, dst, .. } => (vec![src], vec![dst]),
        Instruction::Binary { lhs, rhs, .. } => (vec![lhs, rhs], vec![rhs]),
        Instruction::Unary { src, .. } => (vec![src], vec![src]),
        Instruction::Cmp(_, v1, v2) => (vec![v1, v2], vec![]),
        Instruction::SetCc(_, dst) => (vec![], vec![dst]),
        Instruction::Push(op) => (vec![op], vec![]),
        Instruction::Idiv(_, divisor) => (
            vec![
                divisor,
                &Operand::Reg(Register::Ax),
                &Operand::Reg(Register::Dx),
            ],
            vec![&Operand::Reg(Register::Ax), &Operand::Reg(Register::Dx)],
        ),
        Instruction::Cdq(_) => (
            vec![&Operand::Reg(Register::Ax)],
            vec![&Operand::Reg(Register::Dx)],
        ),
        Instruction::Call(op) => todo!(),
        _ => {
            todo!()
        }
    }
}
