use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Rem, Sub};

use copy_analysis::copy_propagation;
use liveness_analysis::eliminate_dead_stores;

use crate::{
    ast::{Const, VarType},
    control_flow::Cfg,
    semantics::type_check::SymbolTable,
    tacky::{BinaryOp, Instruction, Program, TopLevelItem, UnaryOp, Val},
};

mod copy_analysis;
pub mod egglog;
mod liveness_analysis;

macro_rules! fold_binary {
    ($arg:expr; $($op:pat => ($f:ident, $fd:ident)),*) => {
        match $arg {
            $(
                Instruction::Binary {
                    op: $op,
                    lhs: Val::Constant(lhs),
                    rhs: Val::Constant(rhs),
                    dst,
                } => match (lhs, rhs) {
                    (Const::Char(lhs), Const::Char(rhs)) => {
                        *$arg = Instruction::Copy {
                            src: Val::Constant(Const::Char((*lhs).$f(*rhs))),
                            dst: dst.clone(),
                        };
                    }
                    (Const::UChar(lhs), Const::UChar(rhs)) => {
                        *$arg = Instruction::Copy {
                            src: Val::Constant(Const::UChar((*lhs).$f(*rhs))),
                            dst: dst.clone(),
                        };
                    }
                    (Const::Int(lhs), Const::Int(rhs)) => {
                        *$arg = Instruction::Copy {
                            src: Val::Constant(Const::Int((*lhs).$f(*rhs))),
                            dst: dst.clone(),
                        };
                    }
                    (Const::Uint(lhs), Const::Uint(rhs)) => {
                        *$arg = Instruction::Copy {
                            src: Val::Constant(Const::Uint((*lhs).$f(*rhs))),
                            dst: dst.clone(),
                        };
                    }
                    (Const::Long(lhs), Const::Long(rhs)) => {
                        *$arg = Instruction::Copy {
                            src: Val::Constant(Const::Long((*lhs).$f(*rhs))),
                            dst: dst.clone(),
                        };
                    }
                    (Const::Ulong(lhs), Const::Ulong(rhs)) => {
                        *$arg = Instruction::Copy {
                            src: Val::Constant(Const::Ulong((*lhs).$f(*rhs))),
                            dst: dst.clone(),
                        };
                    }
                    (Const::Double(lhs), Const::Double(rhs)) => {
                        *$arg = Instruction::Copy {
                            src: Val::Constant(Const::Double((*lhs).$fd(*rhs))),
                            dst: dst.clone(),
                        };
                    }
                    _ => {}
                },
            )*
            _ => {}
        }
    };
}

macro_rules! fold_binary_int {
    ($arg:expr; $($op:pat => $f:ident),*) => {
        match $arg {
            $(
                Instruction::Binary {
                    op: $op,
                    lhs: Val::Constant(lhs),
                    rhs: Val::Constant(rhs),
                    dst,
                } => match (lhs, rhs) {
                    (Const::Char(lhs), Const::Char(rhs)) => {
                        *$arg = Instruction::Copy {
                            src: Val::Constant(Const::Char((*lhs).$f(*rhs))),
                            dst: dst.clone(),
                        };
                    }
                    (Const::UChar(lhs), Const::UChar(rhs)) => {
                        *$arg = Instruction::Copy {
                            src: Val::Constant(Const::UChar((*lhs).$f(*rhs))),
                            dst: dst.clone(),
                        };
                    }
                    (Const::Int(lhs), Const::Int(rhs)) => {
                        *$arg = Instruction::Copy {
                            src: Val::Constant(Const::Int((*lhs).$f(*rhs))),
                            dst: dst.clone(),
                        };
                    }
                    (Const::Uint(lhs), Const::Uint(rhs)) => {
                        *$arg = Instruction::Copy {
                            src: Val::Constant(Const::Uint((*lhs).$f(*rhs))),
                            dst: dst.clone(),
                        };
                    }
                    (Const::Long(lhs), Const::Long(rhs)) => {
                        *$arg = Instruction::Copy {
                            src: Val::Constant(Const::Long((*lhs).$f(*rhs))),
                            dst: dst.clone(),
                        };
                    }
                    (Const::Ulong(lhs), Const::Ulong(rhs)) => {
                        *$arg = Instruction::Copy {
                            src: Val::Constant(Const::Ulong((*lhs).$f(*rhs))),
                            dst: dst.clone(),
                        };
                    }
                    _ => {}
                },
            )*
            _ => {}
        }
    };
}

macro_rules! fold_binary_int_shift {
    ($arg:expr; $($op:pat => $f:ident),*) => {
        match $arg {
            $(
                Instruction::Binary {
                    op: $op,
                    lhs: Val::Constant(lhs),
                    rhs: Val::Constant(rhs),
                    dst,
                } => match (lhs, rhs) {
                    (Const::Char(lhs), Const::Char(rhs)) => {
                        *$arg = Instruction::Copy {
                            src: Val::Constant(Const::Char((*lhs).$f(*rhs as u32))),
                            dst: dst.clone(),
                        };
                    }
                    (Const::UChar(lhs), Const::UChar(rhs)) => {
                        *$arg = Instruction::Copy {
                            src: Val::Constant(Const::UChar((*lhs).$f(*rhs as u32))),
                            dst: dst.clone(),
                        };
                    }
                    (Const::Int(lhs), Const::Int(rhs)) => {
                        *$arg = Instruction::Copy {
                            src: Val::Constant(Const::Int((*lhs).$f(*rhs as u32))),
                            dst: dst.clone(),
                        };
                    }
                    (Const::Uint(lhs), Const::Uint(rhs)) => {
                        *$arg = Instruction::Copy {
                            src: Val::Constant(Const::Uint((*lhs).$f(*rhs as u32))),
                            dst: dst.clone(),
                        };
                    }
                    (Const::Long(lhs), Const::Long(rhs)) => {
                        *$arg = Instruction::Copy {
                            src: Val::Constant(Const::Long((*lhs).$f(*rhs as u32))),
                            dst: dst.clone(),
                        };
                    }
                    (Const::Ulong(lhs), Const::Ulong(rhs)) => {
                        *$arg = Instruction::Copy {
                            src: Val::Constant(Const::Ulong((*lhs).$f(*rhs as u32))),
                            dst: dst.clone(),
                        };
                    }
                    _ => {}
                },
            )*
            _ => {}
        }
    };
}

macro_rules! fold_binary_cmp {
    ($arg:expr; $($op:pat => $f:ident),*) => {
        match $arg {
            $(
                Instruction::Binary {
                    op: $op,
                    lhs: Val::Constant(lhs),
                    rhs: Val::Constant(rhs),
                    dst,
                } => match (lhs, rhs) {
                    (Const::Char(lhs), Const::Char(rhs)) => {
                        *$arg = Instruction::Copy {
                            src: Val::Constant(Const::Int((*lhs).$f(rhs) as i32)),
                            dst: dst.clone(),
                        };
                    }
                    (Const::UChar(lhs), Const::UChar(rhs)) => {
                        *$arg = Instruction::Copy {
                            src: Val::Constant(Const::Int((*lhs).$f(rhs) as i32)),
                            dst: dst.clone(),
                        };
                    }
                    (Const::Int(lhs), Const::Int(rhs)) => {
                        *$arg = Instruction::Copy {
                            src: Val::Constant(Const::Int((*lhs).$f(rhs) as i32)),
                            dst: dst.clone(),
                        };
                    }
                    (Const::Uint(lhs), Const::Uint(rhs)) => {
                        *$arg = Instruction::Copy {
                            src: Val::Constant(Const::Int((*lhs).$f(rhs) as i32)),
                            dst: dst.clone(),
                        };
                    }
                    (Const::Long(lhs), Const::Long(rhs)) => {
                        *$arg = Instruction::Copy {
                             src: Val::Constant(Const::Int((*lhs).$f(rhs) as i32)),
                            dst: dst.clone(),
                        };
                    }
                    (Const::Ulong(lhs), Const::Ulong(rhs)) => {
                        *$arg = Instruction::Copy {
                            src: Val::Constant(Const::Int((*lhs).$f(rhs) as i32)),
                            dst: dst.clone(),
                        };
                    }
                    (Const::Double(lhs), Const::Double(rhs)) => {
                        *$arg = Instruction::Copy {
                            src: Val::Constant(Const::Int((*lhs).$f(rhs) as i32)),
                            dst: dst.clone(),
                        };
                    }
                    _ => {}
                },
            )*
            _ => {}
        }
    };
}

pub fn constant_folding(program: &mut [Instruction], symbol_table: &SymbolTable) {
    for inst in program.iter_mut() {
        if let Instruction::Binary {
            op: BinaryOp::Divide | BinaryOp::Remainder,
            lhs: _,
            rhs: Val::Constant(c),
            dst: _,
        } = inst
        {
            if matches!(
                c,
                Const::Int(0) | Const::Uint(0) | Const::Long(0) | Const::Ulong(0)
            ) {
                continue;
            }
        }
        fold_binary!(inst; BinaryOp::Add => (wrapping_add, add), BinaryOp::Subtract => (wrapping_sub, sub), BinaryOp::Multiply => (wrapping_mul, mul), BinaryOp::Divide => (div, div), BinaryOp::Remainder => (rem, rem));
        fold_binary_int!(inst; BinaryOp::BitAnd => bitand, BinaryOp::BitOr => bitor, BinaryOp::BitXor => bitxor);
        fold_binary_int_shift!(inst; BinaryOp::ShiftLeft => wrapping_shl, BinaryOp::ShiftRight => wrapping_shr);
        fold_binary_cmp!(inst; BinaryOp::Equal => eq, BinaryOp::NotEqual => ne, BinaryOp::LessThan => lt, BinaryOp::LessOrEqual => le, BinaryOp::GreaterThan => gt, BinaryOp::GreaterOrEqual => ge);

        if let Instruction::Unary {
            op,
            src: Val::Constant(c),
            dst,
        } = inst
        {
            let val = match op {
                UnaryOp::Negate => match c {
                    Const::Char(c) => Const::Char(-*c),
                    Const::UChar(c) => Const::UChar(!*c + 1),
                    Const::Int(c) => Const::Int(-*c),
                    Const::Uint(c) => Const::Uint(!*c + 1),
                    Const::Long(c) => Const::Long(-*c),
                    Const::Ulong(c) => Const::Ulong(!*c + 1),
                    Const::Double(c) => Const::Double(-*c),
                },
                UnaryOp::Complement => match c {
                    Const::Char(c) => Const::Char(!*c),
                    Const::UChar(c) => Const::UChar(!*c),
                    Const::Int(c) => Const::Int(!*c),
                    Const::Uint(c) => Const::Uint(!*c),
                    Const::Long(c) => Const::Long(!*c),
                    Const::Ulong(c) => Const::Ulong(!*c),
                    _ => panic!(),
                },
                UnaryOp::Not => Const::Int(
                    if c == &Const::Char(0)
                        || c == &Const::UChar(0)
                        || c == &Const::Int(0)
                        || c == &Const::Uint(0)
                        || c == &Const::Long(0)
                        || c == &Const::Ulong(0)
                        || c == &Const::Double(0.0)
                        || c == &Const::Double(-0.0)
                    {
                        1
                    } else {
                        0
                    },
                ),
            };

            *inst = Instruction::Copy {
                src: Val::Constant(val),
                dst: dst.clone(),
            };
        }

        if let Instruction::Cast {
            src: Val::Constant(c),
            dst,
        } = inst
        {
            let src = match dst.ty(symbol_table) {
                VarType::Base(base) => match base {
                    crate::ast::BaseType::Char => Const::Char(c.get_char()),
                    crate::ast::BaseType::SChar => Const::Char(c.get_char()),
                    crate::ast::BaseType::UChar => Const::UChar(c.get_uchar()),
                    crate::ast::BaseType::Int => Const::Int(c.get_int()),
                    crate::ast::BaseType::Long => Const::Long(c.get_long()),
                    crate::ast::BaseType::Uint => Const::Uint(c.get_uint()),
                    crate::ast::BaseType::Ulong => Const::Ulong(c.get_ulong()),
                    crate::ast::BaseType::Double => Const::Double(c.get_double()),
                },
                VarType::Pointer(_) => Const::Ulong(c.get_ulong()),
                _ => panic!(),
            };
            *inst = Instruction::Copy {
                src: Val::Constant(src),
                dst: dst.clone(),
            };
        }

        if let Instruction::JumpIfNotZero {
            src: Val::Constant(c),
            dst,
        } = inst
        {
            if c.is_zero() {
                *inst = Instruction::Nop;
            } else {
                *inst = Instruction::Jump(dst.clone());
            }
        }

        if let Instruction::JumpIfZero {
            src: Val::Constant(c),
            dst,
        } = inst
        {
            if c.is_zero() {
                *inst = Instruction::Jump(dst.clone());
            } else {
                *inst = Instruction::Nop;
            }
        }

        if let Instruction::AddPtr {
            ptr,
            index: Val::Constant(c),
            scale,
            dst,
        } = inst
        {
            if c.is_zero() {
                *inst = Instruction::Copy {
                    src: ptr.clone(),
                    dst: dst.clone(),
                };
            } else {
                *inst = Instruction::AddPtr {
                    ptr: ptr.clone(),
                    index: Val::Constant(Const::Long(c.get_long().wrapping_mul(*scale as _))),
                    scale: 1,
                    dst: dst.clone(),
                };
            }
        }
    }
}

pub enum OptimizeOption {
    ConstantFolding,
    DeadCodeElimination,
    CopyPropagation,
    EliminateDeadStores,
}

pub fn optimize(program: &mut Program, symbol_table: &SymbolTable, options: &[OptimizeOption]) {
    for top in &mut program.top_levels {
        if let TopLevelItem::Function(f) = top {
            loop {
                let snapshot = f.body.clone();

                for opt in options {
                    match opt {
                        OptimizeOption::ConstantFolding => {
                            constant_folding(&mut f.body, symbol_table);
                        }
                        OptimizeOption::DeadCodeElimination => {
                            let mut graph = Cfg::new(&f.body);
                            graph.eliminate_unreachable_code();
                            f.body = graph.program();
                        }
                        OptimizeOption::CopyPropagation => {
                            let mut graph = Cfg::new(&f.body);
                            copy_propagation(&mut graph, symbol_table);
                            f.body = graph.program();
                        }
                        OptimizeOption::EliminateDeadStores => {
                            let mut graph = Cfg::new(&f.body);
                            eliminate_dead_stores(&mut graph, symbol_table);
                            f.body = graph.program();
                        }
                    }
                }
                f.body.retain(|inst| !matches!(inst, Instruction::Nop));

                if snapshot == f.body {
                    break;
                }
            }
        }
    }
}
