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
    Idiv(Operand),
    Cdq,
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
                    src: val_to_operand(val),
                    dst: Operand::Reg(Register::Ax),
                });
                body.push(Instruction::Ret);
            }
            tacky::Instruction::Unary { op, src, dst } => {
                body.push(Instruction::Mov {
                    src: val_to_operand(src),
                    dst: val_to_operand(dst),
                });
                body.push(Instruction::Unary {
                    op: match op {
                        tacky::UnaryOp::Negate => UnaryOp::Neg,
                        tacky::UnaryOp::Complement => UnaryOp::Not,
                    },
                    src: val_to_operand(dst),
                });
            }
            tacky::Instruction::Binary { op, lhs, rhs, dst } => match op {
                tacky::BinaryOp::Add | tacky::BinaryOp::Subtract | tacky::BinaryOp::Multiply => {
                    body.push(Instruction::Mov {
                        src: val_to_operand(lhs),
                        dst: val_to_operand(dst),
                    });
                    body.push(Instruction::Binary {
                        op: match op {
                            tacky::BinaryOp::Add => BinaryOp::Add,
                            tacky::BinaryOp::Subtract => BinaryOp::Sub,
                            tacky::BinaryOp::Multiply => BinaryOp::Mult,
                            _ => unreachable!(),
                        },
                        lhs: val_to_operand(rhs),
                        rhs: val_to_operand(dst),
                    });
                }
                tacky::BinaryOp::Divide => {
                    body.push(Instruction::Mov {
                        src: val_to_operand(lhs),
                        dst: Operand::Reg(Register::Ax),
                    });
                    body.push(Instruction::Cdq);
                    body.push(Instruction::Idiv(val_to_operand(rhs)));
                    body.push(Instruction::Mov {
                        src: Operand::Reg(Register::Ax),
                        dst: val_to_operand(dst),
                    });
                }
                tacky::BinaryOp::Remainder => {
                    body.push(Instruction::Mov {
                        src: val_to_operand(lhs),
                        dst: Operand::Reg(Register::Ax),
                    });
                    body.push(Instruction::Cdq);
                    body.push(Instruction::Idiv(val_to_operand(rhs)));
                    body.push(Instruction::Mov {
                        src: Operand::Reg(Register::Dx),
                        dst: val_to_operand(dst),
                    });
                }
            },
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

fn val_to_operand(val: &tacky::Val) -> Operand {
    match val {
        tacky::Val::Constant(imm) => Operand::Imm(*imm),
        tacky::Val::Var(var) => Operand::Pseudo(var.clone()),
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
            Operand::Reg(reg) => write!(f, "{}", reg)?,
            Operand::Pseudo(_) => panic!("Pseudo operand should have been removed"),
            Operand::Stack(offset) => write!(f, "{}(%rbp)", offset)?,
        }

        Ok(())
    }
}

impl Display for Register {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Register::Ax => write!(f, "%eax")?,
            Register::R10 => write!(f, "%r10d")?,
            Register::Dx => write!(f, "%edx")?,
            Register::R11 => write!(f, "%r11d")?,
        }
        Ok(())
    }
}
