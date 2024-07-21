use std::{collections::HashMap, fmt::Display};

use ecow::EcoString;

use crate::tacky;

#[derive(Debug)]
pub struct Program {
    pub function_definition: Function,
}

#[derive(Debug)]
pub struct Function {
    pub name: EcoString,
    pub body: Vec<Instruction>,
}

#[derive(Debug)]
pub enum Instruction {
    Mov {
        src: Operand,
        dst: Operand,
    },
    Unary {
        op: UnaryOp,
        src: Operand,
    },
    Binary {
        op: BinaryOp,
        lhs: Operand,
        rhs: Operand,
    },
    Cmp(Operand, Operand),
    Idiv(Operand),
    Cdq,
    Jmp(EcoString),
    JmpCc(CondCode, EcoString),
    SetCc(CondCode, Operand),
    Label(EcoString),
    AllocateStack(usize),
    Ret,
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
    Imm(i32),
    Reg(Register),
    Pseudo(EcoString),
    Stack(i32),
}

#[derive(Debug, Clone)]
pub enum Register {
    Ax,
    Dx,
    R10,
    R11,
}

pub enum RegisterSize<'a> {
    Byte(&'a Register),
    Dword(&'a Register),
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

pub fn gen_program(program: &tacky::Program) -> Program {
    Program {
        function_definition: gen_function(&program.function_definition),
    }
}

fn gen_function(function: &tacky::Function) -> Function {
    let mut body = Vec::new();

    for inst in &function.body {
        match inst {
            tacky::Instruction::Return(val) => {
                body.push(Instruction::Mov {
                    src: val.into(),
                    dst: Operand::Reg(Register::Ax),
                });
                body.push(Instruction::Ret);
            }
            tacky::Instruction::Unary { op, src, dst } => {
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
                            src: src.into(),
                            dst: dst.into(),
                        });
                        body.push(Instruction::Unary {
                            op,
                            src: dst.into(),
                        });
                    }
                    Unary::Not => {
                        body.push(Instruction::Cmp(Operand::Imm(0), src.into()));
                        body.push(Instruction::Mov {
                            src: Operand::Imm(0),
                            dst: dst.into(),
                        });
                        body.push(Instruction::SetCc(CondCode::E, dst.into()));
                    }
                }
            }
            tacky::Instruction::Binary { op, lhs, rhs, dst } => match op {
                tacky::BinaryOp::Add | tacky::BinaryOp::Subtract | tacky::BinaryOp::Multiply => {
                    body.push(Instruction::Mov {
                        src: lhs.into(),
                        dst: dst.into(),
                    });
                    body.push(Instruction::Binary {
                        op: match op {
                            tacky::BinaryOp::Add => BinaryOp::Add,
                            tacky::BinaryOp::Subtract => BinaryOp::Sub,
                            tacky::BinaryOp::Multiply => BinaryOp::Mult,
                            _ => unreachable!(),
                        },
                        lhs: rhs.into(),
                        rhs: dst.into(),
                    });
                }
                tacky::BinaryOp::Divide => {
                    body.push(Instruction::Mov {
                        src: lhs.into(),
                        dst: Operand::Reg(Register::Ax),
                    });
                    body.push(Instruction::Cdq);
                    body.push(Instruction::Idiv(rhs.into()));
                    body.push(Instruction::Mov {
                        src: Operand::Reg(Register::Ax),
                        dst: dst.into(),
                    });
                }
                tacky::BinaryOp::Remainder => {
                    body.push(Instruction::Mov {
                        src: lhs.into(),
                        dst: Operand::Reg(Register::Ax),
                    });
                    body.push(Instruction::Cdq);
                    body.push(Instruction::Idiv(rhs.into()));
                    body.push(Instruction::Mov {
                        src: Operand::Reg(Register::Dx),
                        dst: dst.into(),
                    });
                }
                tacky::BinaryOp::Equal
                | tacky::BinaryOp::NotEqual
                | tacky::BinaryOp::LessThan
                | tacky::BinaryOp::LessOrEqual
                | tacky::BinaryOp::GreaterThan
                | tacky::BinaryOp::GreaterOrEqual => {
                    body.push(Instruction::Cmp(rhs.into(), lhs.into()));
                    body.push(Instruction::Mov {
                        src: Operand::Imm(0),
                        dst: dst.into(),
                    });
                    body.push(Instruction::SetCc(
                        match op {
                            tacky::BinaryOp::Equal => CondCode::E,
                            tacky::BinaryOp::NotEqual => CondCode::Ne,
                            tacky::BinaryOp::LessThan => CondCode::L,
                            tacky::BinaryOp::LessOrEqual => CondCode::Le,
                            tacky::BinaryOp::GreaterThan => CondCode::G,
                            tacky::BinaryOp::GreaterOrEqual => CondCode::Ge,
                            _ => unreachable!(),
                        },
                        dst.into(),
                    ));
                }
            },
            tacky::Instruction::Copy { src, dst } => {
                body.push(Instruction::Mov {
                    src: src.into(),
                    dst: dst.into(),
                });
            }
            tacky::Instruction::Jump(label) => {
                body.push(Instruction::Jmp(label.clone()));
            }
            tacky::Instruction::JumpIfZero { src, dst } => {
                body.push(Instruction::Cmp(Operand::Imm(0), src.into()));
                body.push(Instruction::JmpCc(CondCode::E, dst.clone()));
            }
            tacky::Instruction::JumpIfNotZero { src, dst } => {
                body.push(Instruction::Cmp(Operand::Imm(0), src.into()));
                body.push(Instruction::JmpCc(CondCode::Ne, dst.clone()));
            }
            tacky::Instruction::Label(label) => {
                body.push(Instruction::Label(label.clone()));
            }
        }
    }

    let stack_size = pseudo_to_stack(&mut body);
    body.insert(0, Instruction::AllocateStack(stack_size));
    body = avoid_mov_mem_mem(body);

    Function {
        name: function.name.clone(),
        body,
    }
}

impl From<&tacky::Val> for Operand {
    fn from(val: &tacky::Val) -> Self {
        match val {
            tacky::Val::Constant(imm) => Operand::Imm(*imm),
            tacky::Val::Var(var) => Operand::Pseudo(var.clone()),
        }
    }
}

fn pseudo_to_stack(insts: &mut [Instruction]) -> usize {
    let mut known_vars = HashMap::new();

    let mut remove_pseudo = |operand: &mut Operand| {
        if let Operand::Pseudo(var) = operand {
            let rsp = (known_vars.len() as i32 + 1) * -4;
            let offset = known_vars.entry(var.clone()).or_insert(rsp);
            *operand = Operand::Stack(*offset);
        }
    };

    for inst in insts {
        match inst {
            Instruction::Mov { src, dst } => {
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
            Instruction::Cdq => {}
            Instruction::Idiv(op) => {
                remove_pseudo(op);
            }
            Instruction::AllocateStack(_) => {}
            Instruction::Ret => {}
            Instruction::Cmp(lhs, rhs) => {
                remove_pseudo(lhs);
                remove_pseudo(rhs);
            }
            Instruction::Jmp(_) => {}
            Instruction::JmpCc(_, _) => {}
            Instruction::SetCc(_, dst) => {
                remove_pseudo(dst);
            }
            Instruction::Label(_) => {}
        }
    }

    known_vars.len() * 4
}

fn avoid_mov_mem_mem(insts: Vec<Instruction>) -> Vec<Instruction> {
    let mut new_insts = Vec::new();

    for inst in insts {
        match inst {
            Instruction::Mov {
                src: src @ Operand::Stack(_),
                dst: dst @ Operand::Stack(_),
            } => {
                new_insts.push(Instruction::Mov {
                    src,
                    dst: Operand::Reg(Register::R10),
                });
                new_insts.push(Instruction::Mov {
                    src: Operand::Reg(Register::R10),
                    dst,
                });
            }
            Instruction::Idiv(op) if matches!(op, Operand::Imm(_)) => {
                new_insts.push(Instruction::Mov {
                    src: op,
                    dst: Operand::Reg(Register::R10),
                });
                new_insts.push(Instruction::Idiv(Operand::Reg(Register::R10)));
            }
            Instruction::Binary {
                op: op @ (BinaryOp::Add | BinaryOp::Sub),
                lhs,
                rhs,
            } if matches!((&lhs, &rhs), (Operand::Stack(_), Operand::Stack(_))) => {
                new_insts.push(Instruction::Mov {
                    src: lhs,
                    dst: Operand::Reg(Register::R10),
                });
                new_insts.push(Instruction::Binary {
                    op,
                    lhs: Operand::Reg(Register::R10),
                    rhs,
                });
            }
            Instruction::Binary {
                op: BinaryOp::Mult,
                lhs,
                rhs: rhs @ Operand::Stack(_),
            } => {
                new_insts.push(Instruction::Mov {
                    src: rhs.clone(),
                    dst: Operand::Reg(Register::R11),
                });
                new_insts.push(Instruction::Binary {
                    op: BinaryOp::Mult,
                    lhs,
                    rhs: Operand::Reg(Register::R11),
                });
                new_insts.push(Instruction::Mov {
                    src: Operand::Reg(Register::R11),
                    dst: rhs,
                });
            }
            Instruction::Cmp(lhs @ Operand::Stack(_), rhs @ Operand::Stack(_)) => {
                new_insts.push(Instruction::Mov {
                    src: lhs,
                    dst: Operand::Reg(Register::R10),
                });
                new_insts.push(Instruction::Cmp(Operand::Reg(Register::R10), rhs));
            }
            Instruction::Cmp(lhs, rhs @ Operand::Imm(_)) => {
                new_insts.push(Instruction::Mov {
                    src: rhs,
                    dst: Operand::Reg(Register::R11),
                });
                new_insts.push(Instruction::Cmp(lhs, Operand::Reg(Register::R11)));
            }
            _ => new_insts.push(inst),
        }
    }
    new_insts
}

impl Display for Program {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", self.function_definition)?;
        writeln!(f, ".section .note.GNU-stack,\"\",@progbits")?;
        Ok(())
    }
}

impl Display for Function {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, ".globl {}", self.name)?;
        writeln!(f, "{}:", self.name)?;
        writeln!(f, "pushq %rbp")?;
        writeln!(f, "movq %rsp, %rbp")?;
        for inst in &self.body {
            write!(f, "{inst}")?;
        }
        Ok(())
    }
}

impl Display for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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
        }
        Ok(())
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
        }

        Ok(())
    }
}

impl<'a> Display for RegisterSize<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RegisterSize::Byte(reg) => match reg {
                Register::Ax => write!(f, "%al")?,
                Register::Dx => write!(f, "%dl")?,
                Register::R10 => write!(f, "%r10b")?,
                Register::R11 => write!(f, "%r11b")?,
            },
            RegisterSize::Dword(reg) => match reg {
                Register::Ax => write!(f, "%eax")?,
                Register::Dx => write!(f, "%edx")?,
                Register::R10 => write!(f, "%r10d")?,
                Register::R11 => write!(f, "%r11d")?,
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
