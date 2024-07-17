use std::fmt::Display;

use ecow::EcoString;

use crate::ast;

#[derive(Debug)]
pub struct Program {
    pub function_denifition: Function,
}

#[derive(Debug)]
pub struct Function {
    pub name: EcoString,
    pub body: Vec<Instruction>,
}

#[derive(Debug)]
pub enum Instruction {
    Mov { src: Operand, dst: Operand },
    Ret,
}

#[derive(Debug)]
pub enum Operand {
    Imm(i32),
    Register,
}

pub fn gen_program(program: &ast::Program) -> Program {
    Program {
        function_denifition: gen_function(&program.function_definition),
    }
}

fn gen_function(function: &ast::Function) -> Function {
    Function {
        name: function.name.clone(),
        body: statement_inst(&function.body),
    }
}

fn statement_inst(statement: &ast::Statement) -> Vec<Instruction> {
    match statement {
        ast::Statement::Return(ast::Expression::Constant(imm)) => {
            vec![
                Instruction::Mov {
                    src: Operand::Imm(*imm),
                    dst: Operand::Register,
                },
                Instruction::Ret,
            ]
        }
    }
}

impl Display for Program {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", self.function_denifition)?;
        writeln!(f, ".section .note.GNU-stack,\"\",@progbits")?;
        Ok(())
    }
}

impl Display for Function {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, ".globl {}", self.name)?;
        writeln!(f, "{}:", self.name)?;
        for inst in &self.body {
            writeln!(f, "    {}", inst)?;
        }
        Ok(())
    }
}

impl Display for Instruction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Instruction::Mov { src, dst } => {
                writeln!(f, "movl {}, {}", src, dst)
            }
            Instruction::Ret => {
                writeln!(f, "ret")
            }
        }
    }
}

impl Display for Operand {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Operand::Imm(imm) => {
                write!(f, "${}", imm)
            }
            Operand::Register => {
                write!(f, "%eax")
            }
        }
    }
}
