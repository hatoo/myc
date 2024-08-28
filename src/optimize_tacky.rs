use std::ops::{Add, Div, Mul, Rem, Sub};

use crate::{
    ast::Const,
    semantics::type_check::SymbolTable,
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

pub fn constant_folding(program: &mut Vec<Instruction>, symbol_table: &SymbolTable) {
    for inst in program.iter_mut() {
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
                    Const::Char(c) => Const::Char(-*c),
                    Const::Int(c) => Const::Int(-*c),
                    Const::Long(c) => Const::Long(-*c),
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
            let src = if let crate::ast::VarType::Base(base) = dst.ty(symbol_table) {
                match base {
                    crate::ast::BaseType::Char => Const::Char(c.get_char()),
                    crate::ast::BaseType::SChar => Const::Char(c.get_char()),
                    crate::ast::BaseType::UChar => Const::UChar(c.get_uchar()),
                    crate::ast::BaseType::Int => Const::Int(c.get_int()),
                    crate::ast::BaseType::Long => Const::Long(c.get_long()),
                    crate::ast::BaseType::Uint => Const::Uint(c.get_uint()),
                    crate::ast::BaseType::Ulong => Const::Ulong(c.get_ulong()),
                    crate::ast::BaseType::Double => Const::Double(c.get_double()),
                }
            } else {
                panic!()
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
            src: Val::Constant(Const::Int(i)),
            dst,
        } = inst
        {
            if *i != 0 {
                *inst = Instruction::Jump(dst.clone());
            } else {
                *inst = Instruction::Nop;
            }
        }

        if let Instruction::JumpIfZero {
            src: Val::Constant(Const::Int(i)),
            dst,
        } = inst
        {
            if *i == 0 {
                *inst = Instruction::Jump(dst.clone());
            } else {
                *inst = Instruction::Nop;
            }
        }
    }

    program.retain(|inst| !matches!(inst, Instruction::Nop));
}

pub fn optimize(program: &mut Program, symbol_table: &SymbolTable) {
    for top in &mut program.top_levels {
        if let TopLevelItem::Function(f) = top {
            loop {
                let snapshot = f.body.clone();
                constant_folding(&mut f.body, symbol_table);
                if snapshot == f.body {
                    break;
                }
            }
        }
    }
}
