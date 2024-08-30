use std::ops::{Add, Div, Mul, Rem, Sub};

use crate::{
    ast::{Const, VarType},
    semantics::type_check::SymbolTable,
    tacky::{BinaryOp, Instruction, Program, TopLevelItem, UnaryOp, Val},
};

mod copy_analysis;
mod graph;

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

pub fn constant_folding(program: &mut Vec<Instruction>, symbol_table: &SymbolTable) {
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
                UnaryOp::Not => Const::Int(if c.get_ulong() == 0 { 1 } else { 0 }),
            };

            *inst = Instruction::Copy {
                src: Val::Constant(val),
                dst: dst.clone(),
            };
        }

        if let Instruction::DoubleToInt {
            src: Val::Constant(c),
            dst,
        }
        | Instruction::DoubleToUint {
            src: Val::Constant(c),
            dst,
        }
        | Instruction::SignExtend {
            src: Val::Constant(c),
            dst,
        }
        | Instruction::ZeroExtend {
            src: Val::Constant(c),
            dst,
        }
        | Instruction::Truncate {
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

        if let Instruction::IntToDouble {
            src: Val::Constant(c),
            dst,
        }
        | Instruction::UintToDouble {
            src: Val::Constant(c),
            dst,
        } = inst
        {
            let src = Val::Constant(Const::Double(c.get_double()));
            *inst = Instruction::Copy {
                src,
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
    }

    program.retain(|inst| !matches!(inst, Instruction::Nop));
}

pub enum OptimizeOption {
    ConstantFolding,
    DeadCodeElimination,
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
                            let mut graph = graph::Graph::new(&f.body);
                            graph.eliminate_unreachable_code();
                            f.body = graph.program();
                        }
                    }
                }

                if snapshot == f.body {
                    break;
                }
            }
        }
    }
}
