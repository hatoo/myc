use core::panic;
use std::{collections::HashMap, fmt::Display};

use ecow::EcoString;

use crate::{
    ast::{self, Const, VarType},
    semantics::{self, type_check::SymbolTable},
    tacky,
};

#[derive(Debug)]
pub struct Program {
    pub top_levels: Vec<TopLevel>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssemblyType {
    LongWord,
    QuadWord,
    Double,
}

impl AssemblyType {
    pub fn suffix(&self) -> &'static str {
        match self {
            AssemblyType::LongWord => "l",
            AssemblyType::QuadWord => "q",
            AssemblyType::Double => "sd",
        }
    }
}

impl<'a> From<&'a ast::VarType> for AssemblyType {
    fn from(ty: &'a ast::VarType) -> Self {
        match ty {
            ast::VarType::Int => AssemblyType::LongWord,
            ast::VarType::Uint => AssemblyType::LongWord,
            ast::VarType::Ulong => AssemblyType::QuadWord,
            ast::VarType::Long => AssemblyType::QuadWord,
            ast::VarType::Double => AssemblyType::Double,
        }
    }
}

impl From<ast::VarType> for AssemblyType {
    fn from(ty: ast::VarType) -> Self {
        match ty {
            ast::VarType::Int => AssemblyType::LongWord,
            ast::VarType::Uint => AssemblyType::LongWord,
            ast::VarType::Ulong => AssemblyType::QuadWord,
            ast::VarType::Long => AssemblyType::QuadWord,
            ast::VarType::Double => AssemblyType::Double,
        }
    }
}

#[derive(Debug)]
pub enum TopLevel {
    StaticConstant(StaticConstant),
    StaticVariable(StaticVariable),
    Function(Function),
}

#[derive(Debug)]
pub struct StaticConstant {
    pub name: EcoString,
    pub alignment: usize,
    pub init: semantics::type_check::StaticInit,
}

#[derive(Debug)]
pub struct StaticVariable {
    pub global: bool,
    pub name: EcoString,
    pub init: semantics::type_check::StaticInit,
}

#[derive(Debug)]
pub struct Function {
    pub global: bool,
    pub name: EcoString,
    pub body: Vec<Instruction>,
}

#[derive(Debug)]
pub enum Instruction {
    Mov {
        ty: AssemblyType,
        src: Operand,
        dst: Operand,
    },
    Movsx {
        src: Operand,
        dst: Operand,
    },
    MovZeroExtend {
        src: Operand,
        dst: Operand,
    },
    Unary {
        op: UnaryOp,
        ty: AssemblyType,
        src: Operand,
    },
    Binary {
        op: BinaryOp,
        ty: AssemblyType,
        lhs: Operand,
        rhs: Operand,
    },
    Cmp(AssemblyType, Operand, Operand),
    Idiv(AssemblyType, Operand),
    Div(AssemblyType, Operand),
    Cdq(AssemblyType),
    Jmp(EcoString),
    JmpCc(CondCode, EcoString),
    SetCc(CondCode, Operand),
    Label(EcoString),
    Ret,
    Push(Operand),
    Call(EcoString),
    Cvttsd2si {
        ty: AssemblyType,
        src: Operand,
        dst: Operand,
    },
    Cvtsi2sd {
        ty: AssemblyType,
        src: Operand,
        dst: Operand,
    },
}

#[derive(Debug)]
pub enum UnaryOp {
    Neg,
    Not,
    Shr,
}

#[derive(Debug)]
pub enum BinaryOp {
    Add,
    Sub,
    Mult,
    DivDouble,
    And,
    Or,
    Xor,
}

#[derive(Debug, Clone)]
pub enum Pseudo {
    Var(EcoString),
    // Must be placed in read only section
    Double { value: f64, alignment: usize },
}

#[derive(Debug, Clone)]
pub enum Operand {
    Imm(u64),
    Reg(Register),
    Pseudo(Pseudo),
    Stack(i32),
    Data(EcoString),
}

impl Operand {
    fn sized(&self, size: AssemblyType) -> SizedOperand {
        SizedOperand { ty: size, op: self }
    }
}

impl From<tacky::Val> for Operand {
    fn from(val: tacky::Val) -> Self {
        match val {
            tacky::Val::Constant(Const::Double(d)) => Operand::Pseudo(Pseudo::Double {
                value: d,
                alignment: 8,
            }),
            tacky::Val::Constant(imm) => Operand::Imm(imm.get_ulong()),
            tacky::Val::Var(var) => Operand::Pseudo(Pseudo::Var(var)),
        }
    }
}

impl<'a> From<&'a tacky::Val> for Operand {
    fn from(val: &'a tacky::Val) -> Self {
        match val {
            tacky::Val::Constant(Const::Double(d)) => Operand::Pseudo(Pseudo::Double {
                value: *d,
                alignment: 8,
            }),
            tacky::Val::Constant(imm) => Operand::Imm(imm.get_ulong()),
            tacky::Val::Var(var) => Operand::Pseudo(Pseudo::Var(var.clone())),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Register {
    Ax,
    Cx,
    Dx,
    Di,
    Si,
    R8,
    R9,
    R10,
    R11,
    SP,
    Xmm(u8),
}

pub enum RegisterSize<'a> {
    Byte(&'a Register),
    Dword(&'a Register),
    Qword(&'a Register),
}

#[derive(Debug, Clone)]
pub enum CondCode {
    E,
    Ne,
    G,
    Ge,
    L,
    Le,
    A,
    Ae,
    B,
    Be,
}

/*
pub enum AsmEntry {
    Obj { ty: AssemblyType, is_static: bool },
    Fun { is_defined: bool },
}
*/

#[derive(Debug, Default)]
pub struct ConstTable {
    counter: usize,
    table: HashMap<(u64, usize), EcoString>,
}

impl ConstTable {
    fn label(&mut self, double: f64, alignment: usize) -> EcoString {
        let key = (double.to_bits(), alignment);

        match self.table.entry(key) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.get().clone(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                let label = EcoString::from(format!("double{}", self.counter));
                self.counter += 1;
                entry.insert(label.clone());
                label
            }
        }
    }
}

#[derive(Debug)]
pub struct CodeGen<'a> {
    const_table: ConstTable,
    label_counter: usize,
    symbol_table: &'a SymbolTable,
}

impl<'a> CodeGen<'a> {
    pub fn new(symbol_table: &'a SymbolTable) -> Self {
        Self {
            const_table: ConstTable::default(),
            label_counter: 0,
            symbol_table,
        }
    }

    fn gen_label(&mut self, prefix: &str) -> EcoString {
        let label = format!("codegen.{}.{}", prefix, self.label_counter);
        self.label_counter += 1;
        EcoString::from(label)
    }

    pub fn gen_program(&mut self, program: &tacky::Program) -> Program {
        let top_levels: Vec<_> = program
            .top_levels
            .iter()
            .map(|item| match item {
                tacky::TopLevelItem::StaticVariable(tacky::StaticVariable {
                    global,
                    name,
                    init,
                }) => TopLevel::StaticVariable(StaticVariable {
                    global: *global,
                    name: name.clone(),
                    init: *init,
                }),
                tacky::TopLevelItem::Function(function) => {
                    TopLevel::Function(self.gen_function(function))
                }
            })
            .collect();

        Program {
            top_levels: self
                .const_table
                .table
                .iter()
                .map(|((value, align), v)| {
                    TopLevel::StaticConstant(StaticConstant {
                        name: v.clone(),
                        alignment: *align,
                        init: semantics::type_check::StaticInit::Double(f64::from_bits(*value)),
                    })
                })
                .chain(top_levels.into_iter())
                .collect(),
        }
    }
    fn gen_function(&mut self, function: &tacky::Function) -> Function {
        let mut body = Vec::new();

        let semantics::type_check::Attr::Fun { ty, .. } = &self.symbol_table[&function.name] else {
            unreachable!()
        };

        let (int_reg_args, double_reg_args, stack_args) =
            classify_parameters(function.params.iter().zip(ty.params.iter()));

        for (i, (param, ty)) in int_reg_args.into_iter().enumerate() {
            body.push(Instruction::Mov {
                ty: ty.into(),
                src: Operand::Reg(PARAM_REGISTERS[i]),
                dst: Operand::Pseudo(Pseudo::Var(param.clone())),
            });
        }

        for (i, (param, ty)) in double_reg_args.into_iter().enumerate() {
            body.push(Instruction::Mov {
                ty: ty.into(),
                src: Operand::Reg(Register::Xmm(i as _)),
                dst: Operand::Pseudo(Pseudo::Var(param.clone())),
            });
        }

        for (i, (param, ty)) in stack_args.into_iter().enumerate() {
            body.push(Instruction::Mov {
                ty: ty.into(),
                src: Operand::Stack((16 + i * 8) as i32),
                dst: Operand::Pseudo(Pseudo::Var(param.clone())),
            });
        }

        for inst in &function.body {
            match inst {
                tacky::Instruction::Return(val) => {
                    if ty.ret == VarType::Double {
                        body.push(Instruction::Mov {
                            ty: AssemblyType::Double,
                            src: val.into(),
                            dst: Operand::Reg(Register::Xmm(0)),
                        });
                        body.push(Instruction::Ret);
                    } else {
                        body.push(Instruction::Mov {
                            ty: ty.ret.into(),
                            src: val.into(),
                            dst: Operand::Reg(Register::Ax),
                        });
                        body.push(Instruction::Ret);
                    }
                }
                tacky::Instruction::Unary { op, src, dst } => {
                    let src_ty = src.ty(&self.symbol_table);
                    let dst_ty = dst.ty(&self.symbol_table);

                    match (src_ty, op) {
                        (VarType::Double, tacky::UnaryOp::Not) => {
                            body.push(Instruction::Binary {
                                op: BinaryOp::Xor,
                                ty: AssemblyType::Double,
                                lhs: Operand::Reg(Register::Xmm(0)),
                                rhs: Operand::Reg(Register::Xmm(0)),
                            });
                            body.push(Instruction::Cmp(
                                AssemblyType::Double,
                                src.into(),
                                Operand::Reg(Register::Xmm(0)),
                            ));
                            body.push(Instruction::Mov {
                                ty: dst_ty.into(),
                                src: Operand::Imm(0),
                                dst: dst.into(),
                            });
                            body.push(Instruction::SetCc(CondCode::E, dst.into()));
                        }
                        (VarType::Double, tacky::UnaryOp::Negate) => {
                            body.push(Instruction::Mov {
                                ty: AssemblyType::Double,
                                src: src.into(),
                                dst: dst.into(),
                            });
                            body.push(Instruction::Binary {
                                op: BinaryOp::Xor,
                                ty: AssemblyType::Double,
                                lhs: Operand::Pseudo(Pseudo::Double {
                                    value: -0.0,
                                    alignment: 16,
                                }),
                                rhs: dst.into(),
                            });
                        }
                        _ => {
                            enum Unary {
                                Simple(UnaryOp),
                                Not,
                            }

                            let op = match op {
                                tacky::UnaryOp::Negate => Unary::Simple(UnaryOp::Neg),
                                tacky::UnaryOp::Complement => Unary::Simple(UnaryOp::Not),
                                tacky::UnaryOp::Not => Unary::Not,
                            };

                            match op {
                                Unary::Simple(op) => {
                                    body.push(Instruction::Mov {
                                        ty: src.ty(&self.symbol_table).into(),
                                        src: src.into(),
                                        dst: dst.into(),
                                    });
                                    body.push(Instruction::Unary {
                                        ty: src.ty(&self.symbol_table).into(),
                                        op,
                                        src: dst.into(),
                                    });
                                }
                                Unary::Not => {
                                    body.push(Instruction::Cmp(
                                        src.ty(&self.symbol_table).into(),
                                        Operand::Imm(0),
                                        src.into(),
                                    ));
                                    body.push(Instruction::Mov {
                                        ty: dst_ty.into(),
                                        src: Operand::Imm(0),
                                        dst: dst.into(),
                                    });
                                    body.push(Instruction::SetCc(CondCode::E, dst.into()));
                                }
                            }
                        }
                    }
                }
                tacky::Instruction::Binary { op, lhs, rhs, dst } => {
                    enum Binary {
                        Simple(BinaryOp),
                        Divide,
                        Remainder,
                        Compare(CondCode),
                    }

                    let ty = lhs.ty(&self.symbol_table);

                    let op = match op {
                        tacky::BinaryOp::Add => Binary::Simple(BinaryOp::Add),
                        tacky::BinaryOp::Subtract => Binary::Simple(BinaryOp::Sub),
                        tacky::BinaryOp::Multiply => Binary::Simple(BinaryOp::Mult),
                        tacky::BinaryOp::Divide => Binary::Divide,
                        tacky::BinaryOp::Remainder => Binary::Remainder,
                        tacky::BinaryOp::Equal => Binary::Compare(CondCode::E),
                        tacky::BinaryOp::NotEqual => Binary::Compare(CondCode::Ne),
                        tacky::BinaryOp::LessThan => Binary::Compare(if ty.is_signed() {
                            CondCode::L
                        } else {
                            CondCode::B
                        }),
                        tacky::BinaryOp::LessOrEqual => Binary::Compare(if ty.is_signed() {
                            CondCode::Le
                        } else {
                            CondCode::Be
                        }),
                        tacky::BinaryOp::GreaterThan => Binary::Compare(if ty.is_signed() {
                            CondCode::G
                        } else {
                            CondCode::A
                        }),
                        tacky::BinaryOp::GreaterOrEqual => Binary::Compare(if ty.is_signed() {
                            CondCode::Ge
                        } else {
                            CondCode::Ae
                        }),
                    };

                    match op {
                        Binary::Simple(op) => {
                            body.push(Instruction::Mov {
                                ty: lhs.ty(&self.symbol_table).into(),
                                src: lhs.into(),
                                dst: dst.into(),
                            });
                            body.push(Instruction::Binary {
                                ty: lhs.ty(&self.symbol_table).into(),
                                op,
                                lhs: rhs.into(),
                                rhs: dst.into(),
                            });
                        }
                        Binary::Divide => {
                            let ty = lhs.ty(&self.symbol_table);
                            if ty == VarType::Double {
                                let (lhs, rhs) = (rhs, lhs);
                                body.push(Instruction::Mov {
                                    ty: AssemblyType::Double,
                                    src: rhs.into(),
                                    dst: Operand::Reg(Register::Xmm(14)),
                                });
                                body.push(Instruction::Binary {
                                    op: BinaryOp::DivDouble,
                                    ty: AssemblyType::Double,
                                    lhs: lhs.into(),
                                    rhs: Operand::Reg(Register::Xmm(14)),
                                });
                                body.push(Instruction::Mov {
                                    ty: AssemblyType::Double,
                                    src: Operand::Reg(Register::Xmm(14)),
                                    dst: dst.into(),
                                });
                            } else if ty.is_signed() {
                                let ty = ty.into();
                                body.push(Instruction::Mov {
                                    ty,
                                    src: lhs.into(),
                                    dst: Operand::Reg(Register::Ax),
                                });
                                body.push(Instruction::Cdq(ty));
                                body.push(Instruction::Idiv(ty, rhs.into()));
                                body.push(Instruction::Mov {
                                    ty,
                                    src: Operand::Reg(Register::Ax),
                                    dst: dst.into(),
                                });
                            } else {
                                let ty = ty.into();
                                body.push(Instruction::Mov {
                                    ty,
                                    src: lhs.into(),
                                    dst: Operand::Reg(Register::Ax),
                                });
                                body.push(Instruction::Mov {
                                    ty,
                                    src: Operand::Imm(0),
                                    dst: Operand::Reg(Register::Dx),
                                });
                                body.push(Instruction::Div(ty, rhs.into()));
                                body.push(Instruction::Mov {
                                    ty,
                                    src: Operand::Reg(Register::Ax),
                                    dst: dst.into(),
                                });
                            }
                        }
                        Binary::Remainder => {
                            let ty = lhs.ty(&self.symbol_table);
                            if ty.is_signed() {
                                let ty = ty.into();
                                body.push(Instruction::Mov {
                                    ty,
                                    src: lhs.into(),
                                    dst: Operand::Reg(Register::Ax),
                                });
                                body.push(Instruction::Cdq(ty));
                                body.push(Instruction::Idiv(ty, rhs.into()));
                                body.push(Instruction::Mov {
                                    ty,
                                    src: Operand::Reg(Register::Dx),
                                    dst: dst.into(),
                                });
                            } else {
                                let ty = ty.into();
                                body.push(Instruction::Mov {
                                    ty,
                                    src: lhs.into(),
                                    dst: Operand::Reg(Register::Ax),
                                });
                                body.push(Instruction::Mov {
                                    ty,
                                    src: Operand::Imm(0),
                                    dst: Operand::Reg(Register::Dx),
                                });
                                body.push(Instruction::Div(ty, rhs.into()));
                                body.push(Instruction::Mov {
                                    ty,
                                    src: Operand::Reg(Register::Dx),
                                    dst: dst.into(),
                                });
                            }
                        }
                        Binary::Compare(cond) => {
                            body.push(Instruction::Cmp(
                                lhs.ty(&self.symbol_table).into(),
                                rhs.into(),
                                lhs.into(),
                            ));
                            body.push(Instruction::Mov {
                                ty: dst.ty(&self.symbol_table).into(),
                                src: Operand::Imm(0),
                                dst: dst.into(),
                            });
                            body.push(Instruction::SetCc(cond, dst.into()));
                        }
                    }
                }
                tacky::Instruction::Copy { src, dst } => {
                    body.push(Instruction::Mov {
                        ty: src.ty(&self.symbol_table).into(),
                        src: src.into(),
                        dst: dst.into(),
                    });
                }
                tacky::Instruction::Jump(label) => {
                    body.push(Instruction::Jmp(label.clone()));
                }
                tacky::Instruction::JumpIfZero { src, dst } => {
                    if src.ty(&self.symbol_table) == VarType::Double {
                        body.push(Instruction::Binary {
                            op: BinaryOp::Xor,
                            ty: AssemblyType::Double,
                            lhs: Operand::Reg(Register::Xmm(0)),
                            rhs: Operand::Reg(Register::Xmm(0)),
                        });
                        body.push(Instruction::Cmp(
                            AssemblyType::Double,
                            src.into(),
                            Operand::Reg(Register::Xmm(0)),
                        ));
                        body.push(Instruction::JmpCc(CondCode::E, dst.clone()));
                    } else {
                        body.push(Instruction::Cmp(
                            src.ty(&self.symbol_table).into(),
                            Operand::Imm(0),
                            src.into(),
                        ));
                        body.push(Instruction::JmpCc(CondCode::E, dst.clone()));
                    }
                }
                tacky::Instruction::JumpIfNotZero { src, dst } => {
                    if src.ty(&self.symbol_table) == VarType::Double {
                        body.push(Instruction::Binary {
                            op: BinaryOp::Xor,
                            ty: AssemblyType::Double,
                            lhs: Operand::Reg(Register::Xmm(0)),
                            rhs: Operand::Reg(Register::Xmm(0)),
                        });
                        body.push(Instruction::Cmp(
                            AssemblyType::Double,
                            src.into(),
                            Operand::Reg(Register::Xmm(0)),
                        ));
                        body.push(Instruction::JmpCc(CondCode::Ne, dst.clone()));
                    } else {
                        body.push(Instruction::Cmp(
                            src.ty(&self.symbol_table).into(),
                            Operand::Imm(0),
                            src.into(),
                        ));
                        body.push(Instruction::JmpCc(CondCode::Ne, dst.clone()));
                    }
                }
                tacky::Instruction::Label(label) => {
                    body.push(Instruction::Label(label.clone()));
                }
                tacky::Instruction::FunCall { name, args, dst } => {
                    let semantics::type_check::Attr::Fun { ty, .. } = &self.symbol_table[name]
                    else {
                        unreachable!()
                    };

                    let (int_reg_args, double_reg_args, stack_args) =
                        classify_parameters(args.iter().zip(ty.params.iter()));

                    let stack_padding = 8 * (stack_args.len() % 2);

                    if stack_padding > 0 {
                        body.push(Instruction::Binary {
                            op: BinaryOp::Sub,
                            ty: AssemblyType::QuadWord,
                            lhs: Operand::Imm(stack_padding as _),
                            rhs: Operand::Reg(Register::SP),
                        });
                    }

                    for (i, (arg, ty)) in int_reg_args.into_iter().enumerate() {
                        body.push(Instruction::Mov {
                            ty: ty.into(),
                            src: arg.into(),
                            dst: Operand::Reg(PARAM_REGISTERS[i]),
                        });
                    }

                    for (i, (arg, ty)) in double_reg_args.into_iter().enumerate() {
                        body.push(Instruction::Mov {
                            ty: ty.into(),
                            src: arg.into(),
                            dst: Operand::Reg(Register::Xmm(i as _)),
                        });
                    }

                    let stack_len = stack_args.len();
                    for (arg, ty) in stack_args.into_iter().rev() {
                        match ty {
                            VarType::Int | VarType::Uint => {
                                body.push(Instruction::Mov {
                                    ty: AssemblyType::LongWord,
                                    src: arg.into(),
                                    dst: Operand::Reg(Register::Ax),
                                });
                                body.push(Instruction::Push(Operand::Reg(Register::Ax)));
                            }
                            VarType::Long | VarType::Ulong | VarType::Double => {
                                body.push(Instruction::Push(arg.into()));
                            }
                        }
                    }

                    body.push(Instruction::Call(name.clone()));

                    let bytes_to_remove = 8 * stack_len + stack_padding;

                    if bytes_to_remove > 0 {
                        body.push(Instruction::Binary {
                            op: BinaryOp::Add,
                            ty: AssemblyType::QuadWord,
                            lhs: Operand::Imm(bytes_to_remove as _),
                            rhs: Operand::Reg(Register::SP),
                        });
                    }

                    if ty.ret == VarType::Double {
                        body.push(Instruction::Mov {
                            ty: AssemblyType::Double,
                            src: Operand::Reg(Register::Xmm(0)),
                            dst: dst.into(),
                        });
                    } else {
                        body.push(Instruction::Mov {
                            ty: ty.ret.into(),
                            src: Operand::Reg(Register::Ax),
                            dst: dst.into(),
                        });
                    }
                }
                tacky::Instruction::SignExtend { src, dst } => {
                    body.push(Instruction::Movsx {
                        src: src.into(),
                        dst: dst.into(),
                    });
                }
                tacky::Instruction::Truncate { src, dst } => {
                    body.push(Instruction::Mov {
                        ty: AssemblyType::LongWord,
                        src: src.into(),
                        dst: dst.into(),
                    });
                }
                tacky::Instruction::ZeroExtend { src, dst } => {
                    body.push(Instruction::MovZeroExtend {
                        src: src.into(),
                        dst: dst.into(),
                    });
                }
                tacky::Instruction::DoubleToInt { src, dst } => {
                    body.push(Instruction::Cvttsd2si {
                        ty: dst.ty(&self.symbol_table).into(),
                        src: src.into(),
                        dst: dst.into(),
                    });
                }
                tacky::Instruction::DoubleToUint { src, dst } => {
                    if dst.ty(&self.symbol_table) == ast::VarType::Uint {
                        body.push(Instruction::Cvttsd2si {
                            ty: AssemblyType::QuadWord,
                            src: src.into(),
                            dst: Operand::Reg(Register::R10),
                        });
                        body.push(Instruction::Mov {
                            ty: AssemblyType::LongWord,
                            src: Operand::Reg(Register::R10),
                            dst: dst.into(),
                        });
                    } else {
                        let upper_bound = Operand::Pseudo(Pseudo::Double {
                            value: 9223372036854775808.0,
                            alignment: 8,
                        });

                        let ae_upper = self.gen_label("ae_upper");
                        let end = self.gen_label("end");

                        body.push(Instruction::Cmp(
                            AssemblyType::Double,
                            upper_bound.clone(),
                            src.into(),
                        ));
                        body.push(Instruction::JmpCc(CondCode::Ae, ae_upper.clone()));
                        body.push(Instruction::Cvttsd2si {
                            ty: AssemblyType::QuadWord,
                            src: src.into(),
                            dst: dst.into(),
                        });
                        body.push(Instruction::Jmp(end.clone()));
                        body.push(Instruction::Label(ae_upper));
                        body.push(Instruction::Mov {
                            ty: AssemblyType::Double,
                            src: src.into(),
                            dst: Operand::Reg(Register::Xmm(0)),
                        });
                        body.push(Instruction::Binary {
                            op: BinaryOp::Sub,
                            ty: AssemblyType::Double,
                            lhs: upper_bound.clone(),
                            rhs: Operand::Reg(Register::Xmm(0)),
                        });
                        body.push(Instruction::Cvttsd2si {
                            ty: AssemblyType::QuadWord,
                            src: Operand::Reg(Register::Xmm(0)),
                            dst: dst.into(),
                        });
                        body.push(Instruction::Mov {
                            ty: AssemblyType::QuadWord,
                            src: Operand::Imm(9223372036854775808),
                            dst: Operand::Reg(Register::R10),
                        });
                        body.push(Instruction::Binary {
                            op: BinaryOp::Add,
                            ty: AssemblyType::QuadWord,
                            lhs: Operand::Reg(Register::R10),
                            rhs: dst.into(),
                        });
                        body.push(Instruction::Label(end));
                    }
                }
                tacky::Instruction::IntToDouble { src, dst } => {
                    body.push(Instruction::Cvtsi2sd {
                        ty: src.ty(&self.symbol_table).into(),
                        src: src.into(),
                        dst: dst.into(),
                    });
                }
                tacky::Instruction::UintToDouble { src, dst } => match src.ty(&self.symbol_table) {
                    ast::VarType::Uint => {
                        body.push(Instruction::MovZeroExtend {
                            src: src.into(),
                            dst: Operand::Reg(Register::R10),
                        });
                        body.push(Instruction::Cvtsi2sd {
                            ty: AssemblyType::QuadWord,
                            src: Operand::Reg(Register::R10),
                            dst: dst.into(),
                        });
                    }
                    ast::VarType::Ulong => {
                        let l1 = self.gen_label("l1");
                        let end = self.gen_label("end");
                        body.push(Instruction::Cmp(
                            AssemblyType::QuadWord,
                            Operand::Imm(0),
                            src.into(),
                        ));
                        body.push(Instruction::JmpCc(CondCode::L, l1.clone()));
                        body.push(Instruction::Cvtsi2sd {
                            ty: AssemblyType::QuadWord,
                            src: src.into(),
                            dst: dst.into(),
                        });
                        body.push(Instruction::Jmp(end.clone()));
                        body.push(Instruction::Label(l1));
                        body.push(Instruction::Mov {
                            ty: AssemblyType::QuadWord,
                            src: src.into(),
                            dst: Operand::Reg(Register::R10),
                        });
                        body.push(Instruction::Mov {
                            ty: AssemblyType::QuadWord,
                            src: Operand::Reg(Register::R10),
                            dst: Operand::Reg(Register::R11),
                        });
                        body.push(Instruction::Unary {
                            op: UnaryOp::Shr,
                            ty: AssemblyType::QuadWord,
                            src: Operand::Reg(Register::R11),
                        });
                        body.push(Instruction::Binary {
                            op: BinaryOp::And,
                            ty: AssemblyType::QuadWord,
                            lhs: Operand::Imm(1),
                            rhs: Operand::Reg(Register::R10),
                        });
                        body.push(Instruction::Binary {
                            op: BinaryOp::Or,
                            ty: AssemblyType::QuadWord,
                            lhs: Operand::Reg(Register::R10),
                            rhs: Operand::Reg(Register::R11),
                        });
                        body.push(Instruction::Cvtsi2sd {
                            ty: AssemblyType::QuadWord,
                            src: Operand::Reg(Register::R11),
                            dst: dst.into(),
                        });
                        body.push(Instruction::Binary {
                            op: BinaryOp::Add,
                            ty: AssemblyType::Double,
                            lhs: dst.into(),
                            rhs: dst.into(),
                        });
                        body.push(Instruction::Label(end));
                    }
                    _ => unreachable!(),
                },
            }
        }

        let stack_size = pseudo_to_stack(&mut body, &self.symbol_table, &mut self.const_table);
        let stack_size = (stack_size + 15) / 16 * 16;
        body.insert(
            0,
            Instruction::Binary {
                op: BinaryOp::Sub,
                ty: AssemblyType::QuadWord,
                lhs: Operand::Imm(stack_size as _),
                rhs: Operand::Reg(Register::SP),
            },
        );
        body = avoid_mov_mem_mem(body);

        Function {
            global: function.global,
            name: function.name.clone(),
            body,
        }
    }
}

fn classify_parameters<'a, T>(
    iter: impl Iterator<Item = (T, &'a VarType)>,
) -> (Vec<(T, VarType)>, Vec<(T, VarType)>, Vec<(T, VarType)>) {
    let mut int_reg_args = Vec::new();
    let mut double_reg_args = Vec::new();
    let mut stack_args = Vec::new();

    for (param, ty) in iter {
        match ty {
            VarType::Double => {
                if double_reg_args.len() < 8 {
                    double_reg_args.push((param, *ty));
                } else {
                    stack_args.push((param, *ty));
                }
            }
            _ => {
                if int_reg_args.len() < 6 {
                    int_reg_args.push((param, *ty));
                } else {
                    stack_args.push((param, *ty));
                }
            }
        }
    }

    (int_reg_args, double_reg_args, stack_args)
}

const PARAM_REGISTERS: [Register; 6] = [
    Register::Di,
    Register::Si,
    Register::Dx,
    Register::Cx,
    Register::R8,
    Register::R9,
];

fn pseudo_to_stack(
    insts: &mut [Instruction],
    symbol_table: &SymbolTable,
    const_table: &mut ConstTable,
) -> usize {
    let mut total = 0;
    let mut known_vars = HashMap::new();

    let mut remove_pseudo = |operand: &mut Operand| {
        if let Operand::Pseudo(var) = operand {
            match var {
                Pseudo::Var(var) => match &symbol_table[var] {
                    semantics::type_check::Attr::Static { .. } => {
                        *operand = Operand::Data(var.clone());
                    }
                    attr => {
                        if let Some(addr) = known_vars.get(var) {
                            *operand = Operand::Stack(*addr);
                        } else {
                            let size = attr.ty().size() as i32;
                            total += size;
                            total = (total + (size - 1)) / size * size;
                            known_vars.insert(var.clone(), -total);
                            *operand = Operand::Stack(-total);
                        }
                    }
                },
                Pseudo::Double {
                    value: d,
                    alignment,
                } => {
                    *operand = Operand::Data(const_table.label(*d, *alignment));
                }
            }
        }
    };

    for inst in insts {
        match inst {
            Instruction::Mov { ty: _, src, dst } => {
                remove_pseudo(src);
                remove_pseudo(dst);
            }
            Instruction::Unary { src, .. } => {
                remove_pseudo(src);
            }
            Instruction::Binary { lhs, rhs, .. } => {
                remove_pseudo(lhs);
                remove_pseudo(rhs);
            }
            Instruction::Cdq(_) => {}
            Instruction::Idiv(_, op) => {
                remove_pseudo(op);
            }
            Instruction::Ret => {}
            Instruction::Cmp(_, lhs, rhs) => {
                remove_pseudo(lhs);
                remove_pseudo(rhs);
            }
            Instruction::Jmp(_) => {}
            Instruction::JmpCc(_, _) => {}
            Instruction::SetCc(_, dst) => {
                remove_pseudo(dst);
            }
            Instruction::Label(_) => {}
            Instruction::Push(op) => {
                remove_pseudo(op);
            }
            Instruction::Call(_) => {}
            Instruction::Movsx { src, dst } => {
                remove_pseudo(src);
                remove_pseudo(dst);
            }
            Instruction::MovZeroExtend { src, dst } => {
                remove_pseudo(src);
                remove_pseudo(dst);
            }
            Instruction::Div(_, op) => {
                remove_pseudo(op);
            }
            Instruction::Cvttsd2si { ty: _, src, dst } => {
                remove_pseudo(src);
                remove_pseudo(dst);
            }
            Instruction::Cvtsi2sd { ty: _, src, dst } => {
                remove_pseudo(src);
                remove_pseudo(dst);
            }
        }
    }

    total as _
}

fn avoid_mov_mem_mem(insts: Vec<Instruction>) -> Vec<Instruction> {
    let mut new_insts = Vec::new();

    for inst in insts {
        match inst {
            Instruction::Mov {
                ty,
                src: src @ (Operand::Stack(_) | Operand::Data(_)),
                dst: dst @ (Operand::Stack(_) | Operand::Data(_)),
            } => {
                new_insts.push(Instruction::Mov {
                    ty,
                    src,
                    dst: Operand::Reg(if ty == AssemblyType::Double {
                        Register::Xmm(14)
                    } else {
                        Register::R10
                    }),
                });
                new_insts.push(Instruction::Mov {
                    ty,
                    src: Operand::Reg(if ty == AssemblyType::Double {
                        Register::Xmm(14)
                    } else {
                        Register::R10
                    }),
                    dst,
                });
            }
            Instruction::Mov {
                ty: AssemblyType::QuadWord,
                src: src @ Operand::Imm(_),
                dst: dst @ (Operand::Stack(_) | Operand::Data(_)),
            } => {
                new_insts.push(Instruction::Mov {
                    ty: AssemblyType::QuadWord,
                    src,
                    dst: Operand::Reg(Register::R10),
                });
                new_insts.push(Instruction::Mov {
                    ty: AssemblyType::QuadWord,
                    src: Operand::Reg(Register::R10),
                    dst,
                });
            }
            Instruction::Movsx {
                src: src @ Operand::Imm(_),
                dst: dst @ (Operand::Stack(_) | Operand::Data(_)),
            } => {
                new_insts.push(Instruction::Mov {
                    ty: AssemblyType::LongWord,
                    src,
                    dst: Operand::Reg(Register::R10),
                });
                new_insts.push(Instruction::Movsx {
                    src: Operand::Reg(Register::R10),
                    dst: Operand::Reg(Register::R11),
                });
                new_insts.push(Instruction::Mov {
                    ty: AssemblyType::QuadWord,
                    src: Operand::Reg(Register::R11),
                    dst,
                });
            }
            Instruction::Movsx {
                src: src @ Operand::Imm(_),
                dst,
            } => {
                new_insts.push(Instruction::Mov {
                    ty: AssemblyType::LongWord,
                    src,
                    dst: Operand::Reg(Register::R10),
                });
                new_insts.push(Instruction::Movsx {
                    src: Operand::Reg(Register::R10),
                    dst,
                });
            }
            Instruction::Movsx {
                src,
                dst: dst @ (Operand::Stack(_) | Operand::Data(_)),
            } => {
                new_insts.push(Instruction::Movsx {
                    src,
                    dst: Operand::Reg(Register::R10),
                });
                new_insts.push(Instruction::Mov {
                    ty: AssemblyType::QuadWord,
                    src: Operand::Reg(Register::R10),
                    dst,
                });
            }
            Instruction::Idiv(ty, op @ Operand::Imm(_)) => {
                new_insts.push(Instruction::Mov {
                    ty,
                    src: op,
                    dst: Operand::Reg(Register::R10),
                });
                new_insts.push(Instruction::Idiv(ty, Operand::Reg(Register::R10)));
            }
            Instruction::Div(ty, op @ Operand::Imm(_)) => {
                new_insts.push(Instruction::Mov {
                    ty,
                    src: op,
                    dst: Operand::Reg(Register::R10),
                });
                new_insts.push(Instruction::Div(ty, Operand::Reg(Register::R10)));
            }
            Instruction::Binary {
                ty,
                op: op @ (BinaryOp::Add | BinaryOp::Sub | BinaryOp::And | BinaryOp::Or),
                lhs: lhs @ (Operand::Stack(_) | Operand::Data(_)),
                rhs: rhs @ (Operand::Stack(_) | Operand::Data(_)),
            } if !matches!(ty, AssemblyType::Double) => {
                new_insts.push(Instruction::Mov {
                    ty,
                    src: lhs,
                    dst: Operand::Reg(Register::R10),
                });
                new_insts.push(Instruction::Binary {
                    ty,
                    op,
                    lhs: Operand::Reg(Register::R10),
                    rhs,
                });
            }
            Instruction::Binary {
                ty,
                op: BinaryOp::Mult,
                lhs,
                rhs: rhs @ (Operand::Stack(_) | Operand::Data(_)),
            } => {
                let lhs = if ty == AssemblyType::QuadWord && matches!(lhs, Operand::Imm(_)) {
                    new_insts.push(Instruction::Mov {
                        ty,
                        src: lhs,
                        dst: Operand::Reg(Register::R10),
                    });
                    Operand::Reg(Register::R10)
                } else {
                    lhs
                };
                let tmp = if ty == AssemblyType::Double {
                    Operand::Reg(Register::Xmm(15))
                } else {
                    Operand::Reg(Register::R11)
                };
                new_insts.push(Instruction::Mov {
                    ty,
                    src: rhs.clone(),
                    dst: tmp.clone(),
                });
                new_insts.push(Instruction::Binary {
                    ty,
                    op: BinaryOp::Mult,
                    lhs,
                    rhs: tmp.clone(),
                });
                new_insts.push(Instruction::Mov {
                    ty,
                    src: tmp.clone(),
                    dst: rhs,
                });
            }
            Instruction::Binary {
                op,
                ty: AssemblyType::QuadWord,
                lhs,
                rhs,
            } => {
                let lhs = if let Operand::Imm(x) = lhs {
                    if i32::try_from(x).is_err() {
                        new_insts.push(Instruction::Mov {
                            ty: AssemblyType::QuadWord,
                            src: lhs,
                            dst: Operand::Reg(Register::R10),
                        });
                        Operand::Reg(Register::R10)
                    } else {
                        lhs
                    }
                } else {
                    lhs
                };

                let rhs = if let Operand::Imm(x) = rhs {
                    if i32::try_from(x).is_err() {
                        new_insts.push(Instruction::Mov {
                            ty: AssemblyType::QuadWord,
                            src: rhs,
                            dst: Operand::Reg(Register::R11),
                        });
                        Operand::Reg(Register::R11)
                    } else {
                        rhs
                    }
                } else {
                    rhs
                };

                new_insts.push(Instruction::Binary {
                    op,
                    ty: AssemblyType::QuadWord,
                    lhs,
                    rhs,
                });
            }
            Instruction::Binary {
                op,
                ty: AssemblyType::Double,
                lhs,
                rhs,
            } => {
                if !matches!(rhs, Operand::Reg(_)) {
                    new_insts.push(Instruction::Mov {
                        ty: AssemblyType::Double,
                        src: rhs.clone(),
                        dst: Operand::Reg(Register::Xmm(15)),
                    });
                    new_insts.push(Instruction::Binary {
                        op,
                        ty: AssemblyType::Double,
                        lhs,
                        rhs: Operand::Reg(Register::Xmm(15)),
                    });
                    new_insts.push(Instruction::Mov {
                        ty: AssemblyType::Double,
                        src: Operand::Reg(Register::Xmm(15)),
                        dst: rhs,
                    });
                } else {
                    new_insts.push(Instruction::Binary {
                        op,
                        ty: AssemblyType::Double,
                        lhs,
                        rhs,
                    });
                };
            }
            Instruction::Cmp(
                ty,
                lhs @ (Operand::Stack(_) | Operand::Data(_)),
                rhs @ (Operand::Stack(_) | Operand::Data(_)),
            ) if !matches!(ty, AssemblyType::Double) => {
                new_insts.push(Instruction::Mov {
                    ty,
                    src: lhs,
                    dst: Operand::Reg(Register::R10),
                });
                new_insts.push(Instruction::Cmp(ty, Operand::Reg(Register::R10), rhs));
            }
            Instruction::Cmp(ty, lhs, rhs) => match ty {
                AssemblyType::LongWord => {
                    if matches!(rhs, Operand::Imm(_)) {
                        new_insts.push(Instruction::Mov {
                            ty,
                            src: rhs,
                            dst: Operand::Reg(Register::R11),
                        });
                        new_insts.push(Instruction::Cmp(ty, lhs, Operand::Reg(Register::R11)));
                    } else {
                        new_insts.push(Instruction::Cmp(ty, lhs, rhs));
                    }
                }
                AssemblyType::QuadWord => {
                    let lhs = if matches!(lhs, Operand::Imm(_)) {
                        new_insts.push(Instruction::Mov {
                            ty,
                            src: lhs,
                            dst: Operand::Reg(Register::R10),
                        });
                        Operand::Reg(Register::R10)
                    } else {
                        lhs
                    };

                    let rhs = if matches!(rhs, Operand::Imm(_)) {
                        new_insts.push(Instruction::Mov {
                            ty,
                            src: rhs,
                            dst: Operand::Reg(Register::R11),
                        });
                        Operand::Reg(Register::R11)
                    } else {
                        rhs
                    };

                    new_insts.push(Instruction::Cmp(ty, lhs, rhs));
                }
                AssemblyType::Double => {
                    let rhs = if !matches!(rhs, Operand::Reg(_)) {
                        new_insts.push(Instruction::Mov {
                            ty,
                            src: rhs,
                            dst: Operand::Reg(Register::Xmm(15)),
                        });
                        Operand::Reg(Register::Xmm(15))
                    } else {
                        rhs
                    };

                    new_insts.push(Instruction::Cmp(ty, lhs, rhs));
                }
            },
            Instruction::Push(op @ Operand::Imm(_)) => {
                new_insts.push(Instruction::Mov {
                    ty: AssemblyType::QuadWord,
                    src: op,
                    dst: Operand::Reg(Register::R10),
                });
                new_insts.push(Instruction::Push(Operand::Reg(Register::R10)));
            }
            Instruction::MovZeroExtend { src, dst } => {
                if let Operand::Reg(_) = dst {
                    new_insts.push(Instruction::Mov {
                        ty: AssemblyType::LongWord,
                        src,
                        dst,
                    });
                } else {
                    new_insts.push(Instruction::Mov {
                        ty: AssemblyType::LongWord,
                        src,
                        dst: Operand::Reg(Register::R11),
                    });
                    new_insts.push(Instruction::Mov {
                        ty: AssemblyType::QuadWord,
                        src: Operand::Reg(Register::R11),
                        dst,
                    });
                }
            }
            Instruction::Cvttsd2si { ty, src, dst } if !matches!(dst, Operand::Reg(_)) => {
                new_insts.push(Instruction::Cvttsd2si {
                    ty,
                    src,
                    dst: Operand::Reg(Register::R11),
                });
                new_insts.push(Instruction::Mov {
                    ty,
                    src: Operand::Reg(Register::R11),
                    dst,
                });
            }
            Instruction::Cvtsi2sd { ty, src, dst } => {
                let src = if matches!(src, Operand::Imm(_)) {
                    new_insts.push(Instruction::Mov {
                        ty,
                        src,
                        dst: Operand::Reg(Register::R10),
                    });
                    Operand::Reg(Register::R10)
                } else {
                    src
                };

                if !matches!(dst, Operand::Reg(_)) {
                    new_insts.push(Instruction::Cvtsi2sd {
                        ty,
                        src,
                        dst: Operand::Reg(Register::Xmm(15)),
                    });
                    new_insts.push(Instruction::Mov {
                        ty: AssemblyType::Double,
                        src: Operand::Reg(Register::Xmm(15)),
                        dst,
                    });
                } else {
                    new_insts.push(Instruction::Cvtsi2sd { ty, src, dst });
                }
            }
            _ => new_insts.push(inst),
        }
    }
    new_insts
}

struct SizedOperand<'a> {
    ty: AssemblyType,
    op: &'a Operand,
}

impl<'a> Display for SizedOperand<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.op {
            Operand::Imm(imm) => match self.ty {
                // trucated anyway
                AssemblyType::LongWord => write!(f, "${}", *imm as i32)?,
                AssemblyType::QuadWord | AssemblyType::Double => write!(f, "${}", imm)?,
            },
            Operand::Reg(reg) => match self.ty {
                AssemblyType::LongWord => write!(f, "{}", RegisterSize::Dword(reg))?,
                AssemblyType::QuadWord | AssemblyType::Double => {
                    write!(f, "{}", RegisterSize::Qword(reg))?
                }
            },
            Operand::Pseudo(_) => panic!("Pseudo operand should have been removed"),
            Operand::Stack(offset) => write!(f, "{}(%rbp)", offset)?,
            Operand::Data(name) => write!(f, "{}(%rip)", name)?,
        }

        Ok(())
    }
}

impl Display for Program {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for top in &self.top_levels {
            writeln!(f, "{}", top)?;
        }
        writeln!(f, ".section .note.GNU-stack,\"\",@progbits")?;
        Ok(())
    }
}

impl Display for TopLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TopLevel::StaticConstant(constant) => write!(f, "{}", constant)?,
            TopLevel::StaticVariable(var) => write!(f, "{}", var)?,
            TopLevel::Function(func) => write!(f, "{}", func)?,
        }
        Ok(())
    }
}

impl Display for Function {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.global {
            writeln!(f, ".globl {}", self.name)?;
        }
        writeln!(f, ".text")?;
        writeln!(f, "{}:", self.name)?;
        writeln!(f, "pushq %rbp")?;
        writeln!(f, "movq %rsp, %rbp")?;
        for inst in &self.body {
            write!(f, "{inst}")?;
        }
        Ok(())
    }
}

impl Display for StaticConstant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, ".section .rodata")?;
        writeln!(f, ".align {}", self.alignment)?;
        writeln!(f, "{}:", self.name)?;
        match self.init {
            semantics::type_check::StaticInit::Int(x) => writeln!(f, ".long {}", x)?,
            semantics::type_check::StaticInit::Uint(x) => writeln!(f, ".long {}", x)?,
            semantics::type_check::StaticInit::Long(x) => writeln!(f, ".quad {}", x)?,
            semantics::type_check::StaticInit::Ulong(x) => writeln!(f, ".quad {}", x)?,
            semantics::type_check::StaticInit::Double(d) => writeln!(f, ".quad {}", d.to_bits())?,
        }
        Ok(())
    }
}

impl Display for StaticVariable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.global {
            writeln!(f, ".globl {}", self.name)?;
        }
        match self.init {
            semantics::type_check::StaticInit::Int(0)
            | semantics::type_check::StaticInit::Uint(0) => {
                writeln!(f, ".bss")?;
                writeln!(f, ".align {}", self.init.alignment())?;
                writeln!(f, "{}:", self.name)?;
                writeln!(f, ".zero {}", self.init.size())?;
            }
            semantics::type_check::StaticInit::Int(x) => {
                writeln!(f, ".data")?;
                writeln!(f, ".align {}", self.init.alignment())?;
                writeln!(f, "{}:", self.name)?;
                writeln!(f, ".long {}", x)?;
            }
            semantics::type_check::StaticInit::Uint(x) => {
                writeln!(f, ".data")?;
                writeln!(f, ".align {}", self.init.alignment())?;
                writeln!(f, "{}:", self.name)?;
                writeln!(f, ".long {}", x)?;
            }
            semantics::type_check::StaticInit::Long(0)
            | semantics::type_check::StaticInit::Ulong(0) => {
                writeln!(f, ".bss")?;
                writeln!(f, ".align {}", self.init.alignment())?;
                writeln!(f, "{}:", self.name)?;
                writeln!(f, ".zero {}", self.init.size())?;
            }
            semantics::type_check::StaticInit::Long(x) => {
                writeln!(f, ".data")?;
                writeln!(f, ".align {}", self.init.alignment())?;
                writeln!(f, "{}:", self.name)?;
                writeln!(f, ".quad {}", x)?;
            }
            semantics::type_check::StaticInit::Ulong(x) => {
                writeln!(f, ".data")?;
                writeln!(f, ".align {}", self.init.alignment())?;
                writeln!(f, "{}:", self.name)?;
                writeln!(f, ".quad {}", x)?;
            }
            semantics::type_check::StaticInit::Double(d) => {
                writeln!(f, ".data")?;
                writeln!(f, ".align {}", self.init.alignment())?;
                writeln!(f, "{}:", self.name)?;
                writeln!(f, ".quad {}", d.to_bits())?;
            }
        }
        Ok(())
    }
}

impl Display for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Instruction::Mov { ty, src, dst } => {
                writeln!(
                    f,
                    "mov{} {}, {}",
                    ty.suffix(),
                    src.sized(*ty),
                    dst.sized(*ty)
                )?;
            }
            Instruction::Unary { ty, op, src } => {
                writeln!(f, "{op}{} {}", ty.suffix(), src.sized(*ty))?;
            }
            Instruction::Ret => {
                writeln!(f, "movq %rbp, %rsp")?;
                writeln!(f, "popq %rbp")?;
                writeln!(f, "ret")?;
            }
            Instruction::Binary { ty, op, lhs, rhs } => {
                if *ty == AssemblyType::Double {
                    match op {
                        BinaryOp::Mult => {
                            writeln!(
                                f,
                                "mulsd {}, {}",
                                lhs.sized(AssemblyType::Double),
                                rhs.sized(AssemblyType::Double)
                            )?;
                            return Ok(());
                        }
                        BinaryOp::Xor => {
                            writeln!(
                                f,
                                "xorpd {}, {}",
                                lhs.sized(AssemblyType::Double),
                                rhs.sized(AssemblyType::Double)
                            )?;
                            return Ok(());
                        }
                        _ => {}
                    }
                }
                writeln!(
                    f,
                    "{op}{} {}, {}",
                    ty.suffix(),
                    lhs.sized(*ty),
                    rhs.sized(*ty)
                )?;
            }
            Instruction::Cdq(ty) => match ty {
                AssemblyType::LongWord => {
                    writeln!(f, "cdq")?;
                }
                AssemblyType::QuadWord | AssemblyType::Double => {
                    writeln!(f, "cqo")?;
                }
            },
            Instruction::Idiv(ty, op) => {
                writeln!(f, "idiv{} {}", ty.suffix(), op.sized(*ty))?;
            }
            Instruction::Cmp(ty, lhs, rhs) => {
                if *ty == AssemblyType::Double {
                    writeln!(
                        f,
                        "comisd {}, {}",
                        lhs.sized(AssemblyType::Double),
                        rhs.sized(AssemblyType::Double)
                    )?;
                    return Ok(());
                } else {
                    writeln!(
                        f,
                        "cmp{} {}, {}",
                        ty.suffix(),
                        lhs.sized(*ty),
                        rhs.sized(*ty)
                    )?;
                }
            }
            Instruction::Jmp(l) => {
                writeln!(f, "jmp .L{}", l)?;
            }
            Instruction::JmpCc(cond, l) => {
                writeln!(f, "j{} .L{}", cond, l)?;
            }
            Instruction::SetCc(cond, Operand::Reg(reg)) => {
                writeln!(f, "set{} {}", cond, RegisterSize::Byte(reg))?;
            }
            Instruction::SetCc(cond, dst) => {
                writeln!(f, "set{} {}", cond, dst.sized(AssemblyType::LongWord))?;
            }
            Instruction::Label(l) => {
                writeln!(f, ".L{}:", l)?;
            }
            Instruction::Push(op) => {
                writeln!(f, "pushq {}", op.sized(AssemblyType::QuadWord))?;
            }
            Instruction::Call(name) => {
                writeln!(f, "call {}@PLT", name)?;
            }
            Instruction::Movsx { src, dst } => {
                writeln!(
                    f,
                    "movslq {}, {}",
                    src.sized(AssemblyType::LongWord),
                    dst.sized(AssemblyType::QuadWord)
                )?;
            }
            Instruction::MovZeroExtend { .. } => unimplemented!(),
            Instruction::Div(ty, op) => {
                writeln!(f, "div{} {}", ty.suffix(), op.sized(*ty))?;
            }
            Instruction::Cvttsd2si { ty, src, dst } => {
                writeln!(
                    f,
                    "cvttsd2si{} {}, {}",
                    ty.suffix(),
                    src.sized(AssemblyType::Double),
                    dst.sized(*ty)
                )?;
            }
            Instruction::Cvtsi2sd { ty, src, dst } => {
                writeln!(
                    f,
                    "cvtsi2sd{} {}, {}",
                    ty.suffix(),
                    src.sized(*ty),
                    dst.sized(AssemblyType::Double)
                )?;
            }
        }
        Ok(())
    }
}

impl Display for UnaryOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UnaryOp::Neg => write!(f, "neg")?,
            UnaryOp::Not => write!(f, "not")?,
            UnaryOp::Shr => write!(f, "shr")?,
        }
        Ok(())
    }
}

impl Display for BinaryOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BinaryOp::Add => write!(f, "add")?,
            BinaryOp::Sub => write!(f, "sub")?,
            BinaryOp::Mult => write!(f, "imul")?,
            BinaryOp::And => write!(f, "and")?,
            BinaryOp::Or => write!(f, "or")?,
            BinaryOp::DivDouble => write!(f, "div")?,
            BinaryOp::Xor => write!(f, "xor")?,
        }
        Ok(())
    }
}

impl<'a> Display for RegisterSize<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegisterSize::Byte(reg) => match reg {
                Register::Ax => write!(f, "%al")?,
                Register::Cx => write!(f, "%cl")?,
                Register::Dx => write!(f, "%dl")?,
                Register::Di => write!(f, "%dil")?,
                Register::Si => write!(f, "%sil")?,
                Register::R10 => write!(f, "%r10b")?,
                Register::R11 => write!(f, "%r11b")?,
                Register::R8 => write!(f, "%r8b")?,
                Register::R9 => write!(f, "%r9b")?,
                Register::SP => write!(f, "%spl")?,
                Register::Xmm(_) => panic!("XMM registers are only used for QWord"),
            },
            RegisterSize::Dword(reg) => match reg {
                Register::Ax => write!(f, "%eax")?,
                Register::Cx => write!(f, "%ecx")?,
                Register::Dx => write!(f, "%edx")?,
                Register::Di => write!(f, "%edi")?,
                Register::Si => write!(f, "%esi")?,
                Register::R10 => write!(f, "%r10d")?,
                Register::R11 => write!(f, "%r11d")?,
                Register::R8 => write!(f, "%r8d")?,
                Register::R9 => write!(f, "%r9d")?,
                Register::SP => write!(f, "%esp")?,
                Register::Xmm(_) => panic!("XMM registers are only used for QWord"),
            },
            RegisterSize::Qword(reg) => match reg {
                Register::Ax => write!(f, "%rax")?,
                Register::Cx => write!(f, "%rcx")?,
                Register::Dx => write!(f, "%rdx")?,
                Register::Di => write!(f, "%rdi")?,
                Register::Si => write!(f, "%rsi")?,
                Register::R10 => write!(f, "%r10")?,
                Register::R11 => write!(f, "%r11")?,
                Register::R8 => write!(f, "%r8")?,
                Register::R9 => write!(f, "%r9")?,
                Register::SP => write!(f, "%rsp")?,
                Register::Xmm(x) => write!(f, "%xmm{}", x)?,
            },
        }
        Ok(())
    }
}

impl Display for CondCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CondCode::E => write!(f, "e")?,
            CondCode::Ne => write!(f, "ne")?,
            CondCode::G => write!(f, "g")?,
            CondCode::Ge => write!(f, "ge")?,
            CondCode::L => write!(f, "l")?,
            CondCode::Le => write!(f, "le")?,
            CondCode::A => write!(f, "a")?,
            CondCode::Ae => write!(f, "ae")?,
            CondCode::B => write!(f, "b")?,
            CondCode::Be => write!(f, "be")?,
        }
        Ok(())
    }
}
