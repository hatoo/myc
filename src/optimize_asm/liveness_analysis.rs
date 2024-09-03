use std::collections::{HashMap, HashSet};

use ecow::EcoString;

use crate::{
    ast::{BaseType, FunType, VarType},
    codegen::{
        asm_type, classify_struct, is_return_in_memory, Class, CodeGen, Instruction, Operand,
        Pseudo, Register,
    },
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

fn find_used_and_updated<'a>(
    inst: &'a Instruction,
    symbol_table: &SymbolTable,
) -> (Vec<&'a Operand>, Vec<&'a Operand>) {
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
        Instruction::Call(op) => {
            let Operand::Pseudo(Pseudo::Mem { name, .. }) = op else {
                panic!("Do this before pseudo to stack phase")
            };

            let Attr::Fun { ty, .. } = &symbol_table[name] else {
                unreachable!()
            };

            let (i, d, _) = fun_use_registers(ty, symbol_table);

            const INT_REGS: [Operand; 6] = [
                Operand::Reg(Register::Di),
                Operand::Reg(Register::Si),
                Operand::Reg(Register::Dx),
                Operand::Reg(Register::Cx),
                Operand::Reg(Register::R8),
                Operand::Reg(Register::R9),
            ];

            const DOUBLE_REGS: [Operand; 8] = [
                Operand::Reg(Register::Xmm(0)),
                Operand::Reg(Register::Xmm(1)),
                Operand::Reg(Register::Xmm(2)),
                Operand::Reg(Register::Xmm(3)),
                Operand::Reg(Register::Xmm(4)),
                Operand::Reg(Register::Xmm(5)),
                Operand::Reg(Register::Xmm(6)),
                Operand::Reg(Register::Xmm(7)),
            ];

            let used = INT_REGS[..i]
                .iter()
                .chain(DOUBLE_REGS[..d].iter())
                .collect();

            (
                used,
                vec![
                    &Operand::Reg(Register::Di),
                    &Operand::Reg(Register::Si),
                    &Operand::Reg(Register::Dx),
                    &Operand::Reg(Register::Cx),
                    &Operand::Reg(Register::R8),
                    &Operand::Reg(Register::R9),
                    &Operand::Reg(Register::Ax),
                ],
            )
        }
        _ => (vec![], vec![]),
    }
}

fn fun_use_registers(ty: &FunType, symbol_table: &SymbolTable) -> (usize, usize, usize) {
    let mut int_regs = 0;
    let mut double_regs = 0;
    let mut stack_args = 0;

    let return_in_memory = is_return_in_memory(&ty.ret, symbol_table);

    if return_in_memory {
        int_regs += 1;
    }

    let int_regs_available = 6;

    for ty in &ty.params {
        let asm_ty = asm_type(ty, symbol_table);
        match &ty {
            VarType::Base(BaseType::Double) => {
                if double_regs < 8 {
                    double_regs += 1;
                } else {
                    stack_args += 1;
                }
            }
            VarType::Struct(name) => {
                let structure = symbol_table.struct_def(name);
                let classes = classify_struct(structure, &symbol_table);
                let mut use_stack = true;
                let struct_size = structure.size;

                if classes[0] != Class::Memory {
                    let mut tentative_ints = 0;
                    let mut tentative_doubles = 0;
                    let mut offset = 0;
                    for &class in &classes {
                        if class == Class::Sse {
                            tentative_doubles += 1;
                        } else {
                            tentative_ints += 1;
                        }

                        offset += 8;
                    }

                    if (tentative_doubles + double_regs) <= 8
                        && (tentative_ints + int_regs) <= int_regs_available
                    {
                        double_regs += tentative_doubles;
                        int_regs += tentative_ints;
                        use_stack = false;
                    }
                }
                if use_stack {
                    stack_args += classes.len();
                }
            }
            _ => {
                if int_regs < int_regs_available {
                    int_regs += 1;
                } else {
                    stack_args += 1;
                }
            }
        }
    }

    (int_regs, double_regs, stack_args)
}
