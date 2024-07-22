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
    Unary {
        op: UnaryOp,
        src: Val,
        dst: Val,
    },
    Binary {
        op: BinaryOp,
        lhs: Val,
        rhs: Val,
        dst: Val,
    },
    Copy {
        src: Val,
        dst: Val,
    },
    Jump(EcoString),
    JumpIfZero {
        src: Val,
        dst: EcoString,
    },
    JumpIfNotZero {
        src: Val,
        dst: EcoString,
    },
    Label(EcoString),
}

#[derive(Debug)]
pub enum UnaryOp {
    Negate,
    Complement,
    Not,
}

#[derive(Debug)]
pub enum BinaryOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    Remainder,
    Equal,
    NotEqual,
    LessThan,
    LessOrEqual,
    GreaterThan,
    GreaterOrEqual,
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

    fn new_label(&mut self, prefix: &str) -> EcoString {
        let label = EcoString::from(format!("{}.{}", prefix, self.var_counter));
        self.var_counter += 1;
        label
    }

    fn add_statement(&mut self, statement: &ast::Statement) {
        match statement {
            ast::Statement::Return(expression) => {
                let val = self.add_expression(expression);
                self.instructions.push(Instruction::Return(val));
            }
            _ => todo!(),
        }
    }

    fn add_expression(&mut self, expression: &ast::Expression) -> Val {
        match expression {
            ast::Expression::Constant(imm) => Val::Constant(*imm),
            ast::Expression::Unary { op, exp } => {
                let src = self.add_expression(exp);
                let dst = self.new_var();
                self.instructions.push(Instruction::Unary {
                    op: match op {
                        ast::UnaryOp::Negate => UnaryOp::Negate,
                        ast::UnaryOp::Complement => UnaryOp::Complement,
                        ast::UnaryOp::Not => UnaryOp::Not,
                    },
                    src,
                    dst: dst.clone(),
                });
                dst
            }
            ast::Expression::Binary {
                op: ast::BinaryOp::And,
                lhs,
                rhs,
            } => {
                let lhs = self.add_expression(lhs);
                let dst = self.new_var();
                let and_false = self.new_label("and_false");
                self.instructions.push(Instruction::JumpIfZero {
                    src: lhs.clone(),
                    dst: and_false.clone(),
                });
                let rhs = self.add_expression(rhs);
                self.instructions.push(Instruction::JumpIfZero {
                    src: rhs.clone(),
                    dst: and_false.clone(),
                });
                self.instructions.push(Instruction::Copy {
                    src: Val::Constant(1),
                    dst: dst.clone(),
                });
                let end = self.new_label("and_end");
                self.instructions.push(Instruction::Jump(end.clone()));
                self.instructions
                    .push(Instruction::Label(and_false.clone()));
                self.instructions.push(Instruction::Copy {
                    src: Val::Constant(0),
                    dst: dst.clone(),
                });
                self.instructions.push(Instruction::Label(end));
                dst
            }
            ast::Expression::Binary {
                op: ast::BinaryOp::Or,
                lhs,
                rhs,
            } => {
                let lhs = self.add_expression(lhs);
                let dst = self.new_var();
                let or_true = self.new_label("or_true");
                self.instructions.push(Instruction::JumpIfNotZero {
                    src: lhs.clone(),
                    dst: or_true.clone(),
                });
                let rhs = self.add_expression(rhs);
                self.instructions.push(Instruction::JumpIfNotZero {
                    src: rhs.clone(),
                    dst: or_true.clone(),
                });
                self.instructions.push(Instruction::Copy {
                    src: Val::Constant(0),
                    dst: dst.clone(),
                });
                let end = self.new_label("or_end");
                self.instructions.push(Instruction::Jump(end.clone()));
                self.instructions.push(Instruction::Label(or_true.clone()));
                self.instructions.push(Instruction::Copy {
                    src: Val::Constant(1),
                    dst: dst.clone(),
                });
                self.instructions.push(Instruction::Label(end));
                dst
            }
            ast::Expression::Binary { op, lhs, rhs } => {
                let lhs = self.add_expression(lhs);
                let rhs = self.add_expression(rhs);
                let dst = self.new_var();
                self.instructions.push(Instruction::Binary {
                    op: match op {
                        ast::BinaryOp::Add => BinaryOp::Add,
                        ast::BinaryOp::Subtract => BinaryOp::Subtract,
                        ast::BinaryOp::Multiply => BinaryOp::Multiply,
                        ast::BinaryOp::Divide => BinaryOp::Divide,
                        ast::BinaryOp::Remainder => BinaryOp::Remainder,
                        ast::BinaryOp::Equal => BinaryOp::Equal,
                        ast::BinaryOp::NotEqual => BinaryOp::NotEqual,
                        ast::BinaryOp::LessThan => BinaryOp::LessThan,
                        ast::BinaryOp::LessOrEqual => BinaryOp::LessOrEqual,
                        ast::BinaryOp::GreaterThan => BinaryOp::GreaterThan,
                        ast::BinaryOp::GreaterOrEqual => BinaryOp::GreaterOrEqual,
                        _ => unreachable!(),
                    },
                    lhs,
                    rhs,
                    dst: dst.clone(),
                });
                dst
            }
            _ => todo!(),
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
    // generator.add_statement(&function.body);
    Function {
        name: function.name.clone(),
        body: generator.instructions,
    }
}
