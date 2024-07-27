use std::collections::HashMap;

use ecow::EcoString;

use crate::{
    ast::{self, Block},
    semantics,
    span::Spanned,
};

#[derive(Debug)]
pub struct Program {
    pub top_levels: Vec<TopLevelItem>,
}

#[derive(Debug)]
pub enum TopLevelItem {
    Function(Function),
    StaticVariable(StaticVariable),
}

#[derive(Debug)]
pub struct Function {
    pub global: bool,
    pub name: EcoString,
    pub params: Vec<EcoString>,
    pub body: Vec<Instruction>,
}

#[derive(Debug)]
pub struct StaticVariable {
    pub global: bool,
    pub name: EcoString,
    pub init: i32,
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
    FunCall {
        name: EcoString,
        args: Vec<Val>,
        dst: Val,
    },
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

    fn add_block_item(&mut self, block_item: &ast::BlockItem) {
        match block_item {
            ast::BlockItem::Declaration(decl) => match decl {
                ast::Declaration::VarDecl(decl) => {
                    self.add_var_declaration(decl);
                }
                ast::Declaration::FunDecl(_) => {}
            },
            ast::BlockItem::Statement(stmt) => {
                self.add_statement(stmt);
            }
        }
    }

    fn add_var_declaration(&mut self, decl: &ast::VarDecl) {
        if let Some(exp) = &decl.exp {
            let val = self.add_expression(exp);
            self.instructions.push(Instruction::Copy {
                src: val,
                dst: Val::Var(decl.ident.data.clone()),
            });
        }
    }

    fn add_for_init(&mut self, init: &ast::ForInit) {
        match init {
            ast::ForInit::VarDecl(decl) => self.add_var_declaration(decl),
            ast::ForInit::Expression(exp) => {
                self.add_expression(exp);
            }
        }
    }

    fn add_statement(&mut self, statement: &ast::Statement) {
        match statement {
            ast::Statement::Return(expression) => {
                let val = self.add_expression(expression);
                self.instructions.push(Instruction::Return(val));
            }
            ast::Statement::Expression(exp) => {
                self.add_expression(exp);
            }
            ast::Statement::Null => {}
            ast::Statement::If {
                condition,
                then_branch,
                else_branch,
            } => {
                if let Some(else_branch) = else_branch {
                    let else_label = self.new_label("if_else");
                    let end_label = self.new_label("if_end");
                    let cond = self.add_expression(condition);
                    self.instructions.push(Instruction::JumpIfZero {
                        src: cond,
                        dst: else_label.clone(),
                    });
                    self.add_statement(then_branch);
                    self.instructions.push(Instruction::Jump(end_label.clone()));
                    self.instructions.push(Instruction::Label(else_label));
                    self.add_statement(else_branch);
                    self.instructions.push(Instruction::Label(end_label));
                } else {
                    let end_label = self.new_label("if_end");
                    let cond = self.add_expression(condition);
                    self.instructions.push(Instruction::JumpIfZero {
                        src: cond,
                        dst: end_label.clone(),
                    });
                    self.add_statement(then_branch);
                    self.instructions.push(Instruction::Label(end_label));
                }
            }
            ast::Statement::Compound(Block(items)) => {
                for block_item in items {
                    self.add_block_item(block_item);
                }
            }
            ast::Statement::Break { label, .. } => {
                self.instructions
                    .push(Instruction::Jump(format!("break_{}", label).into()));
            }
            ast::Statement::Continue { label, .. } => {
                self.instructions
                    .push(Instruction::Jump(format!("continue_{}", label).into()));
            }
            ast::Statement::DoWhile {
                label,
                condition,
                body,
            } => {
                let start_label: EcoString = format!("do_{}", label).into();
                self.instructions
                    .push(Instruction::Label(start_label.clone()));
                self.add_statement(body);
                self.instructions
                    .push(Instruction::Label(format!("continue_{}", label).into()));

                let cond = self.add_expression(condition);
                self.instructions.push(Instruction::JumpIfNotZero {
                    src: cond,
                    dst: start_label,
                });
                self.instructions
                    .push(Instruction::Label(format!("break_{}", label).into()));
            }
            ast::Statement::While {
                label,
                condition,
                body,
            } => {
                self.instructions
                    .push(Instruction::Label(format!("continue_{}", label).into()));
                let cond = self.add_expression(condition);
                self.instructions.push(Instruction::JumpIfZero {
                    src: cond,
                    dst: format!("break_{}", label).into(),
                });
                self.add_statement(body);
                self.instructions
                    .push(Instruction::Jump(format!("continue_{}", label).into()));
                self.instructions
                    .push(Instruction::Label(format!("break_{}", label).into()));
            }
            ast::Statement::For {
                label,
                init,
                condition,
                step,
                body,
            } => {
                if let Some(init) = init {
                    self.add_for_init(init);
                }
                self.instructions
                    .push(Instruction::Label(format!("start_{}", label).into()));

                if let Some(condition) = condition {
                    let cond = self.add_expression(condition);
                    self.instructions.push(Instruction::JumpIfZero {
                        src: cond,
                        dst: format!("break_{}", label).into(),
                    });
                }

                self.add_statement(body);
                self.instructions
                    .push(Instruction::Label(format!("continue_{}", label).into()));

                if let Some(step) = step {
                    self.add_expression(step);
                }
                self.instructions
                    .push(Instruction::Jump(format!("start_{}", label).into()));
                self.instructions
                    .push(Instruction::Label(format!("break_{}", label).into()));
            }
        }
    }

    fn add_expression(&mut self, expression: &ast::Expression) -> Val {
        match expression {
            ast::Expression::Constant(Spanned { data: imm, .. }) => Val::Constant(*imm),
            ast::Expression::Unary {
                op: Spanned { data: op, .. },
                exp,
            } => {
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
            ast::Expression::Var(Spanned { data: var, .. }) => Val::Var(var.clone()),
            ast::Expression::Assignment { lhs, rhs } => {
                if let ast::Expression::Var(Spanned { data: var, .. }) = lhs.as_ref() {
                    let rhs = self.add_expression(rhs);
                    self.instructions.push(Instruction::Copy {
                        src: rhs,
                        dst: Val::Var(var.clone()),
                    });
                    Val::Var(var.clone())
                } else {
                    panic!("invalid lvalue");
                }
            }
            ast::Expression::Conditional {
                condition,
                then_branch,
                else_branch,
            } => {
                let dst = self.new_var();
                let else_label = self.new_label("cond_else");
                let end_label = self.new_label("cond_end");
                let cond = self.add_expression(condition);
                self.instructions.push(Instruction::JumpIfZero {
                    src: cond,
                    dst: else_label.clone(),
                });
                let v1 = self.add_expression(then_branch);
                self.instructions.push(Instruction::Copy {
                    src: v1,
                    dst: dst.clone(),
                });
                self.instructions.push(Instruction::Jump(end_label.clone()));
                self.instructions.push(Instruction::Label(else_label));
                let v2 = self.add_expression(else_branch);
                self.instructions.push(Instruction::Copy {
                    src: v2,
                    dst: dst.clone(),
                });
                self.instructions.push(Instruction::Label(end_label));
                dst
            }
            ast::Expression::FunctionCall { name, args } => {
                let dst = self.new_var();
                let args = args
                    .iter()
                    .map(|arg| self.add_expression(arg))
                    .collect::<Vec<_>>();
                self.instructions.push(Instruction::FunCall {
                    name: name.data.clone(),
                    args,
                    dst: dst.clone(),
                });
                dst
            }
        }
    }
}

pub fn gen_program(
    program: &ast::Program,
    symbol_table: &HashMap<EcoString, semantics::type_check::Attr>,
) -> Program {
    let mut generator = InstructionGenerator::new();
    Program {
        top_levels: symbol_table
            .iter()
            .filter_map(|(key, value)| {
                if let semantics::type_check::Attr::Static { init, global } = value {
                    let init = match init {
                        semantics::type_check::InitialValue::Initial(i) => *i,
                        semantics::type_check::InitialValue::Tentative => 0,
                        semantics::type_check::InitialValue::NoInitializer => return None,
                    };
                    Some(TopLevelItem::StaticVariable(StaticVariable {
                        global: *global,
                        name: key.clone(),
                        init,
                    }))
                } else {
                    None
                }
            })
            .chain(
                program
                    .decls
                    .iter()
                    .filter_map(|f| match f {
                        ast::Declaration::FunDecl(f) => gen_function(&mut generator, f),
                        _ => None,
                    })
                    .map(TopLevelItem::Function),
            )
            .collect(),
    }
}

fn gen_function(generator: &mut InstructionGenerator, function: &ast::FunDecl) -> Option<Function> {
    if let Some(block) = &function.body {
        for block_item in &block.0 {
            generator.add_block_item(block_item);
        }
        generator.add_statement(&ast::Statement::Return(ast::Expression::Constant(
            Spanned {
                data: 0,
                span: 0..0,
            },
        )));
        Some(Function {
            global: function.storage_class != Some(ast::StorageClass::Static),
            name: function.name.data.clone(),
            params: function.params.iter().map(|s| s.data.clone()).collect(),
            body: std::mem::take(&mut generator.instructions),
        })
    } else {
        None
    }
}
