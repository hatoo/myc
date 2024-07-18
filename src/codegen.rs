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
