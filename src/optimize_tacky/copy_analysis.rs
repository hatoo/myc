use std::collections::{HashMap, HashSet};

use crate::{
    semantics::type_check::SymbolTable,
    tacky::{Instruction, Val},
};

use super::graph::{Node, NodeId};

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub struct Copy {
    pub src: Val,
    pub dst: Val,
}

struct Annotation {
    incoming_copies: HashMap<usize, HashSet<Copy>>,
    annotated_instructions: HashMap<usize, Vec<HashSet<Copy>>>,
}

impl Annotation {
    fn init_block(&mut self, block: &Node) {
        self.annotated_instructions
            .insert(block.id, vec![HashSet::new(); block.instructions.len()]);
    }

    fn annotate_instruction(
        &mut self,
        block_id: usize,
        inst_index: usize,
        reaching_copies: HashSet<Copy>,
    ) {
        self.annotated_instructions.get_mut(&block_id).unwrap()[inst_index] = reaching_copies;
    }

    fn transfer(
        &mut self,
        block: &Node,
        symbol_table: &SymbolTable,
        initial_reaching_copies: HashSet<Copy>,
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
                Instruction::FunCall { dst, .. } => {
                    current_reaching_copies.retain(|c| {
                        !(c.src.is_static(symbol_table)
                            || c.dst.is_static(symbol_table)
                            || Some(&c.src) == dst.as_ref()
                            || Some(&c.dst) == dst.as_ref())
                    });
                }
                Instruction::Unary { dst, .. } | Instruction::Binary { dst, .. } => {
                    current_reaching_copies.retain(|c| !(&c.src == dst || &c.dst == dst));
                }
                _ => {}
            }
        }

        self.incoming_copies
            .insert(block.id, initial_reaching_copies);
    }

    fn meet(&mut self, block: &Node, all_copies: HashSet<Copy>) -> HashSet<Copy> {
        let mut incoming_copies = all_copies.clone();

        for pred in &block.predecessors {
            if let NodeId::Block(id) = pred {
                let pred_copies = self.incoming_copies.get(id).unwrap();
                incoming_copies = incoming_copies.intersection(pred_copies).cloned().collect();
            }
        }

        incoming_copies
    }
}
