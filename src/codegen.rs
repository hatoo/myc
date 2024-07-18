use std::collections::{HashMap, HashSet};

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
    Mov { src: Operand, dst: Operand },
    Unary { op: UnaryOp, src: Operand },
    AllocateStack(usize),
    Ret,
}

#[derive(Debug)]
pub enum UnaryOp {
    Neg,
    Not,
}

#[derive(Debug)]
pub enum Operand {
    Imm(i32),
    Reg(Register),
    Pseudo(EcoString),
    Stack(i32),
}

#[derive(Debug)]
pub enum Register {
    Ax,
    R10,
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

    let mut remove_pseudo = |operand: &mut Operand| match operand {
        Operand::Pseudo(var) => {
            let rsp = (known_vars.len() as i32 + 1) * -4;
            let offset = known_vars.entry(var.clone()).or_insert(rsp);
            *operand = Operand::Stack(*offset);
        }
        _ => {}
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
            Instruction::Mov { src, dst }
            // ????
                if matches!((&src, &dst), (Operand::Stack(_), Operand::Stack(_))) =>
            {
                new_insts.push(Instruction::Mov {
                    src,
                    dst: Operand::Reg(Register::R10),
                });
                new_insts.push(Instruction::Mov {
                    src: Operand::Reg(Register::R10),
                    dst,
                });
            }
            _ => new_insts.push(inst),
        }
    }
    new_insts
}
