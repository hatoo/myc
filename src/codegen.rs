use std::{arch::x86_64::__m128, collections::HashMap, fmt::Display};

use ecow::EcoString;

use crate::{
    ast::{self, VarType},
    semantics, tacky,
};

#[derive(Debug)]
pub struct Program {
    pub top_levels: Vec<TopLevel>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssemblyType {
    LongWord,
    QuadWord,
}

impl<'a> From<&'a ast::VarType> for AssemblyType {
    fn from(ty: &'a ast::VarType) -> Self {
        match ty {
            ast::VarType::Int => AssemblyType::LongWord,
            ast::VarType::Long => AssemblyType::QuadWord,
        }
    }
}

impl From<ast::VarType> for AssemblyType {
    fn from(ty: ast::VarType) -> Self {
        match ty {
            ast::VarType::Int => AssemblyType::LongWord,
            ast::VarType::Long => AssemblyType::QuadWord,
        }
    }
}

#[derive(Debug)]
pub enum TopLevel {
    StaticVariable(StaticVariable),
    Function(Function),
}

#[derive(Debug)]
pub struct StaticVariable {
    pub global: bool,
    pub name: EcoString,
    pub init: semantics::type_check::StaticInit,
    pub alignment: usize,
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
    Cdq(AssemblyType),
    Jmp(EcoString),
    JmpCc(CondCode, EcoString),
    SetCc(CondCode, Operand),
    Label(EcoString),
    Ret,
    Push(Operand),
    Call(EcoString),
}

#[derive(Debug)]
pub enum UnaryOp {
    Neg,
    Not,
}

#[derive(Debug)]
pub enum BinaryOp {
    Add,
    Sub,
    Mult,
}

#[derive(Debug, Clone)]
pub enum Operand {
    Imm(i64),
    Reg(Register),
    Pseudo(EcoString),
    Stack(i32),
    Data(EcoString),
}

impl From<tacky::Val> for Operand {
    fn from(val: tacky::Val) -> Self {
        match val {
            tacky::Val::Constant(imm) => Operand::Imm(imm.get_long()),
            tacky::Val::Var(var) => Operand::Pseudo(var),
        }
    }
}

impl<'a> From<&'a tacky::Val> for Operand {
    fn from(val: &'a tacky::Val) -> Self {
        match val {
            tacky::Val::Constant(imm) => Operand::Imm(imm.get_long()),
            tacky::Val::Var(var) => Operand::Pseudo(var.clone()),
        }
    }
}

#[derive(Debug, Clone)]
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
}

pub fn gen_program(
    program: &tacky::Program,
    symbol_table: &HashMap<EcoString, semantics::type_check::Attr>,
) -> Program {
    Program {
        top_levels: program
            .top_levels
            .iter()
            .map(|item| match item {
                tacky::TopLevelItem::StaticVariable(tacky::StaticVariable {
                    ty,
                    global,
                    name,
                    init,
                }) => TopLevel::StaticVariable(StaticVariable {
                    global: *global,
                    name: name.clone(),
                    init: *init,
                    alignment: match ty {
                        VarType::Int => 4,
                        VarType::Long => 8,
                    },
                }),
                tacky::TopLevelItem::Function(function) => {
                    TopLevel::Function(gen_function(function, symbol_table))
                }
            })
            .collect(),
    }
}

fn gen_function(
    function: &tacky::Function,
    symbol_table: &HashMap<EcoString, semantics::type_check::Attr>,
) -> Function {
    let mut body = Vec::new();

    let semantics::type_check::Attr::Fun { ty, .. } = &symbol_table[&function.name] else {
        unreachable!()
    };

    for (i, (param, ty)) in function.params.iter().zip(ty.params.iter()).enumerate() {
        match i {
            0 => {
                body.push(Instruction::Mov {
                    ty: ty.into(),
                    src: Operand::Reg(Register::Di),
                    dst: Operand::Pseudo(param.clone()),
                });
            }
            1 => {
                body.push(Instruction::Mov {
                    ty: ty.into(),
                    src: Operand::Reg(Register::Si),
                    dst: Operand::Pseudo(param.clone()),
                });
            }
            2 => {
                body.push(Instruction::Mov {
                    ty: ty.into(),
                    src: Operand::Reg(Register::Dx),
                    dst: Operand::Pseudo(param.clone()),
                });
            }
            3 => {
                body.push(Instruction::Mov {
                    ty: ty.into(),
                    src: Operand::Reg(Register::Cx),
                    dst: Operand::Pseudo(param.clone()),
                });
            }
            4 => {
                body.push(Instruction::Mov {
                    ty: ty.into(),
                    src: Operand::Reg(Register::R8),
                    dst: Operand::Pseudo(param.clone()),
                });
            }
            5 => {
                body.push(Instruction::Mov {
                    ty: ty.into(),
                    src: Operand::Reg(Register::R9),
                    dst: Operand::Pseudo(param.clone()),
                });
            }
            _ => {
                body.push(Instruction::Mov {
                    ty: ty.into(),
                    src: Operand::Stack((16 + (i - 6) * 8) as i32),
                    dst: Operand::Pseudo(param.clone()),
                });
            }
        }
    }

    for inst in &function.body {
        match inst {
            tacky::Instruction::Return(val) => {
                body.push(Instruction::Mov {
                    ty: ty.ret.into(),
                    src: val.into(),
                    dst: Operand::Reg(Register::Ax),
                });
                body.push(Instruction::Ret);
            }
            tacky::Instruction::Unary { op, src, dst } => {
                let dst_ty = dst.ty(symbol_table);

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
                            ty: src.ty(symbol_table).into(),
                            src: src.into(),
                            dst: dst.into(),
                        });
                        body.push(Instruction::Unary {
                            ty: src.ty(symbol_table).into(),
                            op,
                            src: dst.into(),
                        });
                    }
                    Unary::Not => {
                        body.push(Instruction::Cmp(
                            src.ty(symbol_table).into(),
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
            tacky::Instruction::Binary { op, lhs, rhs, dst } => {
                enum Binary {
                    Simple(BinaryOp),
                    Divide,
                    Remainder,
                    Compare(CondCode),
                }

                let op = match op {
                    tacky::BinaryOp::Add => Binary::Simple(BinaryOp::Add),
                    tacky::BinaryOp::Subtract => Binary::Simple(BinaryOp::Sub),
                    tacky::BinaryOp::Multiply => Binary::Simple(BinaryOp::Mult),
                    tacky::BinaryOp::Divide => Binary::Divide,
                    tacky::BinaryOp::Remainder => Binary::Remainder,
                    tacky::BinaryOp::Equal => Binary::Compare(CondCode::E),
                    tacky::BinaryOp::NotEqual => Binary::Compare(CondCode::Ne),
                    tacky::BinaryOp::LessThan => Binary::Compare(CondCode::L),
                    tacky::BinaryOp::LessOrEqual => Binary::Compare(CondCode::Le),
                    tacky::BinaryOp::GreaterThan => Binary::Compare(CondCode::G),
                    tacky::BinaryOp::GreaterOrEqual => Binary::Compare(CondCode::Ge),
                };

                match op {
                    Binary::Simple(op) => {
                        body.push(Instruction::Mov {
                            ty: lhs.ty(symbol_table).into(),
                            src: lhs.into(),
                            dst: dst.into(),
                        });
                        body.push(Instruction::Binary {
                            ty: lhs.ty(symbol_table).into(),
                            op,
                            lhs: rhs.into(),
                            rhs: dst.into(),
                        });
                    }
                    Binary::Divide => {
                        body.push(Instruction::Mov {
                            ty: lhs.ty(symbol_table).into(),
                            src: lhs.into(),
                            dst: Operand::Reg(Register::Ax),
                        });
                        body.push(Instruction::Cdq(lhs.ty(symbol_table).into()));
                        body.push(Instruction::Idiv(lhs.ty(symbol_table).into(), rhs.into()));
                        body.push(Instruction::Mov {
                            ty: lhs.ty(symbol_table).into(),
                            src: Operand::Reg(Register::Ax),
                            dst: dst.into(),
                        });
                    }
                    Binary::Remainder => {
                        body.push(Instruction::Mov {
                            ty: lhs.ty(symbol_table).into(),
                            src: lhs.into(),
                            dst: Operand::Reg(Register::Ax),
                        });
                        body.push(Instruction::Cdq(lhs.ty(symbol_table).into()));
                        body.push(Instruction::Idiv(lhs.ty(symbol_table).into(), rhs.into()));
                        body.push(Instruction::Mov {
                            ty: lhs.ty(symbol_table).into(),
                            src: Operand::Reg(Register::Dx),
                            dst: dst.into(),
                        });
                    }
                    Binary::Compare(cond) => {
                        body.push(Instruction::Cmp(
                            lhs.ty(symbol_table).into(),
                            rhs.into(),
                            lhs.into(),
                        ));
                        body.push(Instruction::Mov {
                            ty: dst.ty(symbol_table).into(),
                            src: Operand::Imm(0),
                            dst: dst.into(),
                        });
                        body.push(Instruction::SetCc(cond, dst.into()));
                    }
                }
            }
            tacky::Instruction::Copy { src, dst } => {
                body.push(Instruction::Mov {
                    ty: src.ty(symbol_table).into(),
                    src: src.into(),
                    dst: dst.into(),
                });
            }
            tacky::Instruction::Jump(label) => {
                body.push(Instruction::Jmp(label.clone()));
            }
            tacky::Instruction::JumpIfZero { src, dst } => {
                body.push(Instruction::Cmp(
                    src.ty(symbol_table).into(),
                    Operand::Imm(0),
                    src.into(),
                ));
                body.push(Instruction::JmpCc(CondCode::E, dst.clone()));
            }
            tacky::Instruction::JumpIfNotZero { src, dst } => {
                body.push(Instruction::Cmp(
                    src.ty(symbol_table).into(),
                    Operand::Imm(0),
                    src.into(),
                ));
                body.push(Instruction::JmpCc(CondCode::Ne, dst.clone()));
            }
            tacky::Instruction::Label(label) => {
                body.push(Instruction::Label(label.clone()));
            }
            tacky::Instruction::FunCall { name, args, dst } => {
                let stack_padding = if args.len() > 6 {
                    8 * ((args.len() - 6) % 2)
                } else {
                    0
                };

                if stack_padding > 0 {
                    body.push(Instruction::Binary {
                        op: BinaryOp::Sub,
                        ty: AssemblyType::QuadWord,
                        lhs: Operand::Imm(stack_padding as _),
                        rhs: Operand::Reg(Register::SP),
                    });
                }

                let semantics::type_check::Attr::Fun { ty, .. } = &symbol_table[name] else {
                    unreachable!()
                };

                for (i, (arg, ty)) in args[..std::cmp::min(args.len(), 6)]
                    .iter()
                    .zip(ty.params.iter())
                    .enumerate()
                {
                    match i {
                        0 => body.push(Instruction::Mov {
                            ty: ty.into(),
                            src: arg.into(),
                            dst: Operand::Reg(Register::Di),
                        }),
                        1 => body.push(Instruction::Mov {
                            ty: ty.into(),
                            src: arg.into(),
                            dst: Operand::Reg(Register::Si),
                        }),
                        2 => body.push(Instruction::Mov {
                            ty: ty.into(),
                            src: arg.into(),
                            dst: Operand::Reg(Register::Dx),
                        }),
                        3 => body.push(Instruction::Mov {
                            ty: ty.into(),
                            src: arg.into(),
                            dst: Operand::Reg(Register::Cx),
                        }),
                        4 => body.push(Instruction::Mov {
                            ty: ty.into(),
                            src: arg.into(),
                            dst: Operand::Reg(Register::R8),
                        }),
                        5 => body.push(Instruction::Mov {
                            ty: ty.into(),
                            src: arg.into(),
                            dst: Operand::Reg(Register::R9),
                        }),
                        _ => {
                            unreachable!();
                        }
                    }
                }

                if args.len() > 6 {
                    for arg in args[6..].iter().rev() {
                        match arg.ty(symbol_table) {
                            ast::VarType::Int => {
                                body.push(Instruction::Mov {
                                    ty: AssemblyType::LongWord,
                                    src: arg.into(),
                                    dst: Operand::Reg(Register::Ax),
                                });
                                body.push(Instruction::Push(Operand::Reg(Register::Ax)));
                            }
                            ast::VarType::Long => {
                                body.push(Instruction::Push(arg.into()));
                            }
                        }
                    }
                }
                body.push(Instruction::Call(name.clone()));

                let bytes_to_remove =
                    8 * (if args.len() > 6 { args.len() - 6 } else { 0 }) + stack_padding;

                if bytes_to_remove > 0 {
                    body.push(Instruction::Binary {
                        op: BinaryOp::Add,
                        ty: AssemblyType::QuadWord,
                        lhs: Operand::Imm(bytes_to_remove as _),
                        rhs: Operand::Reg(Register::SP),
                    });
                }

                body.push(Instruction::Mov {
                    ty: dst.ty(symbol_table).into(),
                    src: Operand::Reg(Register::Ax),
                    dst: dst.into(),
                });
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
        }
    }

    let stack_size = pseudo_to_stack(&mut body, symbol_table);
    let stack_size = (stack_size + 15) / 16 * 16;
    body.insert(
        0,
        Instruction::Binary {
            op: BinaryOp::Add,
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

fn pseudo_to_stack(
    insts: &mut [Instruction],
    symbol_table: &HashMap<EcoString, semantics::type_check::Attr>,
) -> usize {
    let mut total = 0;
    let mut known_vars = HashMap::new();

    let mut remove_pseudo = |operand: &mut Operand| {
        if let Operand::Pseudo(var) = operand {
            match &symbol_table[var] {
                semantics::type_check::Attr::Static { .. } => {
                    *operand = Operand::Data(var.clone());
                }
                attr => {
                    if let Some(addr) = known_vars.get(var) {
                        *operand = Operand::Stack(*addr);
                    } else {
                        total -= attr.ty().size() as i32;
                        known_vars.insert(var.clone(), total);
                        *operand = Operand::Stack(total);
                    }
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
        }
    }

    (-total) as _
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
                    dst: Operand::Reg(Register::R10),
                });
                new_insts.push(Instruction::Mov {
                    ty,
                    src: Operand::Reg(Register::R10),
                    dst,
                });
            }
            Instruction::Mov {
                ty: AssemblyType::QuadWord,
                src: src @ Operand::Imm(_),
                dst: dst @ (Operand::Stack(_) | Operand::Data(_)),
            } => {
                new_insts.push(Instruction::Mov {
                    ty: AssemblyType::LongWord,
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
            Instruction::Binary {
                ty,
                op: op @ (BinaryOp::Add | BinaryOp::Sub),
                lhs: lhs @ (Operand::Stack(_) | Operand::Data(_)),
                rhs: rhs @ (Operand::Stack(_) | Operand::Data(_)),
            } => {
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
                new_insts.push(Instruction::Mov {
                    ty,
                    src: rhs.clone(),
                    dst: Operand::Reg(Register::R11),
                });
                new_insts.push(Instruction::Binary {
                    ty,
                    op: BinaryOp::Mult,
                    lhs,
                    rhs: Operand::Reg(Register::R11),
                });
                new_insts.push(Instruction::Mov {
                    ty,
                    src: Operand::Reg(Register::R11),
                    dst: rhs,
                });
            }
            Instruction::Cmp(
                ty,
                lhs @ (Operand::Stack(_) | Operand::Data(_)),
                rhs @ (Operand::Stack(_) | Operand::Data(_)),
            ) => {
                new_insts.push(Instruction::Mov {
                    ty,
                    src: lhs,
                    dst: Operand::Reg(Register::R10),
                });
                new_insts.push(Instruction::Cmp(ty, Operand::Reg(Register::R10), rhs));
            }
            Instruction::Cmp(ty, lhs, rhs @ Operand::Imm(_)) => {
                new_insts.push(Instruction::Mov {
                    ty,
                    src: rhs,
                    dst: Operand::Reg(Register::R11),
                });
                new_insts.push(Instruction::Cmp(ty, lhs, Operand::Reg(Register::R11)));
            }
            _ => new_insts.push(inst),
        }
    }
    new_insts
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

impl Display for StaticVariable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        todo!()
        /*
        if self.global {
            writeln!(f, ".globl {}", self.name)?;
        }
        if self.init == 0 {
            writeln!(f, ".bss")?;
            writeln!(f, ".align 4")?;
            writeln!(f, "{}:", self.name)?;
            writeln!(f, ".zero 4")?;
        } else {
            writeln!(f, ".data")?;
            writeln!(f, "{}:", self.name)?;
            writeln!(f, ".long {}", self.init)?;
        }
        Ok(())
        */
    }
}

impl Display for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        todo!()
        /*
        match self {
            Instruction::Mov { src, dst } => {
                writeln!(f, "movl {src}, {dst}")?;
            }
            Instruction::Unary { op, src } => {
                writeln!(f, "{op} {src}")?;
            }
            Instruction::AllocateStack(n) => {
                writeln!(f, "subq ${n}, %rsp")?;
            }
            Instruction::Ret => {
                writeln!(f, "movq %rbp, %rsp")?;
                writeln!(f, "popq %rbp")?;
                writeln!(f, "ret")?;
            }
            Instruction::Binary { op, lhs, rhs } => {
                writeln!(f, "{op} {lhs}, {rhs}")?;
            }
            Instruction::Cdq => {
                writeln!(f, "cdq")?;
            }
            Instruction::Idiv(op) => {
                writeln!(f, "idivl {op}")?;
            }
            Instruction::Cmp(lhs, rhs) => {
                writeln!(f, "cmpl {lhs}, {rhs}")?;
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
                writeln!(f, "set{} {dst}", cond)?;
            }
            Instruction::Label(l) => {
                writeln!(f, ".L{}:", l)?;
            }
            Instruction::DeallocateStack(n) => {
                writeln!(f, "addq ${n}, %rsp")?;
            }
            Instruction::Push(op) => match op {
                Operand::Imm(imm) => {
                    writeln!(f, "pushq ${}", imm)?;
                }
                Operand::Reg(reg) => {
                    writeln!(f, "pushq {}", RegisterSize::Qword(reg))?;
                }
                _ => unimplemented!(),
            },
            Instruction::Call(name) => {
                writeln!(f, "call {}@PLT", name)?;
            }
        }
        Ok(())
        */
    }
}

impl Display for UnaryOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            UnaryOp::Neg => write!(f, "negl")?,
            UnaryOp::Not => write!(f, "notl")?,
        }
        Ok(())
    }
}

impl Display for BinaryOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BinaryOp::Add => write!(f, "addl")?,
            BinaryOp::Sub => write!(f, "subl")?,
            BinaryOp::Mult => write!(f, "imull")?,
        }
        Ok(())
    }
}

impl Display for Operand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Operand::Imm(imm) => write!(f, "${}", imm)?,
            Operand::Reg(reg) => write!(f, "{}", RegisterSize::Dword(reg))?,
            Operand::Pseudo(_) => panic!("Pseudo operand should have been removed"),
            Operand::Stack(offset) => write!(f, "{}(%rbp)", offset)?,
            Operand::Data(name) => write!(f, "{}(%rip)", name)?,
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
        }
        Ok(())
    }
}
