use ecow::EcoString;

use crate::ast;

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
    Return(Val),
    Unary { op: UnaryOp, src: Val, dst: Val },
}

#[derive(Debug)]
pub enum UnaryOp {
    Negate,
    Complement,
}

#[derive(Debug, Clone)]
pub enum Val {
    Constant(i32),
    Var(EcoString),
}

struct InstructionGenerator {
    var_counter: usize,
    instructions: Vec<Instruction>,
}

impl InstructionGenerator {
    fn new() -> Self {
        Self {
            var_counter: 0,
            instructions: Vec::new(),
        }
    }

    fn new_var(&mut self) -> Val {
        let var = EcoString::from(format!("tmp.{}", self.var_counter));
        self.var_counter += 1;
        Val::Var(var)
    }

    fn add_statement(&mut self, statement: &ast::Statement) {
        match statement {
            ast::Statement::Return(expression) => {
                let val = self.add_expression(expression);
                self.instructions.push(Instruction::Return(val));
            }
        }
    }

    fn add_expression(&mut self, expression: &ast::Expression) -> Val {
        match expression {
            ast::Expression::Constant(imm) => Val::Constant(*imm),
            ast::Expression::Unary(unary) => {
                let src = self.add_expression(&unary.exp);
                let dst = self.new_var();
                self.instructions.push(Instruction::Unary {
                    op: match unary.op {
                        ast::UnaryOp::Negate => UnaryOp::Negate,
                        ast::UnaryOp::Complement => UnaryOp::Complement,
                    },
                    src,
                    dst: dst.clone(),
                });
                dst
            }
        }
    }
}

pub fn gen_program(program: &ast::Program) -> Program {
    Program {
        function_definition: gen_function(&program.function_definition),
    }
}

fn gen_function(function: &ast::Function) -> Function {
    let mut generator = InstructionGenerator::new();
    generator.add_statement(&function.body);
    Function {
        name: function.name.clone(),
        body: generator.instructions,
    }
}
