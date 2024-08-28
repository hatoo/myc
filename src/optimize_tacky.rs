use std::ops::{Add, Div, Mul, Rem, Sub};

use crate::{
    ast::Const,
    tacky::{BinaryOp, Instruction, Program, TopLevelItem, UnaryOp, Val},
};

macro_rules! fold_binary {
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
                    (Const::Double(lhs), Const::Double(rhs)) => {
                        *$arg = Instruction::Copy {
                            src: Val::Constant(Const::Double((*lhs).$f(*rhs))),
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

pub fn constant_folding(program: &mut [Instruction]) {
    for inst in program {
        fold_binary!(inst; BinaryOp::Add => add, BinaryOp::Subtract => sub, BinaryOp::Multiply => mul, BinaryOp::Divide => div, BinaryOp::Remainder => rem);
        fold_binary_cmp!(inst; BinaryOp::Equal => eq, BinaryOp::NotEqual => ne, BinaryOp::LessThan => lt, BinaryOp::LessOrEqual => le, BinaryOp::GreaterThan => gt, BinaryOp::GreaterOrEqual => ge);

        if let Instruction::Unary {
            op,
            src: Val::Constant(c),
            dst,
        } = inst
        {
            let val = match op {
                UnaryOp::Negate => match c {
                    Const::Char(c) => Const::Char(!*c),
                    Const::UChar(c) => Const::UChar(!*c),
                    Const::Int(c) => Const::Int(!*c),
                    Const::Uint(c) => Const::Uint(!*c),
                    Const::Long(c) => Const::Long(!*c),
                    Const::Ulong(c) => Const::Ulong(!*c),
                    _ => panic!(),
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
                UnaryOp::Not => {
                    if let Const::Int(i) = c {
                        Const::Int(if i == &0 { 1 } else { 0 })
                    } else {
                        panic!()
                    }
                }
            };

            *inst = Instruction::Copy {
                src: Val::Constant(val),
                dst: dst.clone(),
            };
        }
    }
}

pub fn optimize(program: &mut Program) {
    for top in &mut program.top_levels {
        if let TopLevelItem::Function(f) = top {
            loop {
                let snapshot = f.body.clone();
                constant_folding(&mut f.body);
                if snapshot == f.body {
                    break;
                }
            }
        }
    }
}
