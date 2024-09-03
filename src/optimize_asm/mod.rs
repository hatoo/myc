use std::collections::{HashMap, HashSet};

use ecow::EcoString;

use crate::{
    codegen::{Instruction, Operand, Pseudo, Register},
    semantics::type_check::{Attr, SymbolTable},
};

mod liveness_analysis;

const FREE_REGISTERS: [Register; 11] = [
    Register::Ax,
    Register::Bx,
    Register::Cx,
    Register::Dx,
    Register::Di,
    Register::Si,
    Register::R8,
    Register::R9,
    Register::R13,
    Register::R14,
    Register::R15,
];

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum NodeId {
    Register(Register),
    Pseudo(EcoString),
}

#[derive(Debug, Clone)]
struct Node {
    id: NodeId,
    neighbors: Vec<NodeId>,
    spill_cost: f32,
    color: Option<usize>,
    pruned: bool,
}

struct Graph {
    map: HashMap<NodeId, Node>,
}

impl Node {
    fn new(id: NodeId) -> Self {
        Self {
            id,
            neighbors: Vec::new(),
            spill_cost: 0.0,
            color: None,
            pruned: false,
        }
    }
}

pub(crate) fn is_scalar(op: &Operand, symbol_table: &SymbolTable) -> Option<NodeId> {
    match op {
        Operand::Pseudo(Pseudo::Mem { name, .. }) => {
            if let Attr::Local(ty) = &symbol_table[name] {
                if ty.is_scalar() {
                    return Some(NodeId::Pseudo(name.clone()));
                }
            }
        }
        Operand::Reg(reg) => {
            if FREE_REGISTERS.contains(reg) || matches!(reg, Register::Xmm(_)) {
                return Some(NodeId::Register(*reg));
            }
        }
        _ => {}
    }

    None
}

impl Graph {
    fn new(program: &[Instruction], symbol_table: &SymbolTable) -> Self {
        todo!()
    }

    fn base() -> Self {
        let mut map = HashMap::new();

        for &reg in &FREE_REGISTERS {
            map.insert(NodeId::Register(reg), Node::new(NodeId::Register(reg)));
        }

        for &reg in &FREE_REGISTERS {
            for &neighbor in &FREE_REGISTERS {
                if reg != neighbor {
                    map.get_mut(&NodeId::Register(reg))
                        .unwrap()
                        .neighbors
                        .push(NodeId::Register(neighbor));
                }
            }
        }

        Self { map }
    }

    fn collect_pseudo_vars(&mut self, insts: &[Instruction], symbol_table: &SymbolTable) {
        let mut vars = HashSet::new();

        let mut add_op = |op: &Operand| {
            if let Some(id) = is_scalar(op, &symbol_table) {
                vars.insert(id);
            }
        };

        for inst in insts {
            match inst {
                Instruction::Binary { lhs, rhs, .. } => {
                    add_op(lhs);
                    add_op(rhs);
                }
                Instruction::Call(op) => {
                    add_op(op);
                }
                Instruction::Cdq(_) => {}
                Instruction::Cmp(_, op1, op2) => {
                    add_op(op1);
                    add_op(op2);
                }
                Instruction::Cvtsi2sd { src, dst, .. } => {
                    add_op(src);
                    add_op(dst);
                }
                Instruction::Cvttsd2si { src, dst, .. } => {
                    add_op(src);
                    add_op(dst);
                }
                Instruction::Div(_, op) => {
                    add_op(op);
                }
                Instruction::Idiv(_, op) => {
                    add_op(op);
                }
                Instruction::Jmp(_) => {}
                Instruction::JmpCc(..) => {}
                Instruction::Label(_) => {}
                Instruction::Lea { src: _, dst } => {
                    // add_op(src);
                    add_op(dst);
                }
                Instruction::Mov { src, dst, .. } => {
                    add_op(src);
                    add_op(dst);
                }
                Instruction::MovZeroExtend { src, dst, .. } => {
                    add_op(src);
                    add_op(dst);
                }
                Instruction::Movsx { src, dst, .. } => {
                    add_op(src);
                    add_op(dst);
                }
                Instruction::Pop(_) => {}
                Instruction::Push(op) => {
                    add_op(op);
                }
                Instruction::Ret => {}
                Instruction::SetCc(_, op) => {
                    add_op(op);
                }
                Instruction::Unary { src, .. } => {
                    add_op(src);
                }
            }
        }

        for id in vars {
            self.map.insert(id.clone(), Node::new(id));
        }
    }
}
