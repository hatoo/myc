use std::collections::HashMap;

use ecow::EcoString;

use crate::{
    ast::{self, Block, VarType},
    semantics::{
        self,
        type_check::{Attr, SymbolTable},
    },
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
    pub init: semantics::type_check::StaticInit,
}

#[derive(Debug)]
pub enum Instruction {
    SignExtend {
        src: Val,
        dst: Val,
    },
    ZeroExtend {
        src: Val,
        dst: Val,
    },
    DoubleToInt {
        src: Val,
        dst: Val,
    },
    DoubleToUint {
        src: Val,
        dst: Val,
    },
    IntToDouble {
        src: Val,
        dst: Val,
    },
    UintToDouble {
        src: Val,
        dst: Val,
    },
    Truncate {
        src: Val,
        dst: Val,
    },
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
    GetAddress {
        src: Val,
        dst: Val,
    },
    Load {
        src: Val,
        dst: Val,
    },
    Store {
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
    Constant(ast::Const),
    Var(EcoString),
}

impl Val {
    pub fn ty(&self, symbol_table: &SymbolTable) -> ast::VarType {
        match self {
            Val::Constant(c) => match c {
                ast::Const::Int(_) => ast::VarType::Int,
                ast::Const::Long(_) => ast::VarType::Long,
                ast::Const::Uint(_) => ast::VarType::Uint,
                ast::Const::Ulong(_) => ast::VarType::Ulong,
                ast::Const::Double(_) => ast::VarType::Double,
            },
            Val::Var(var) => symbol_table[var].ty().clone(),
        }
    }
}

struct InstructionGenerator<'a> {
    var_counter: usize,
    instructions: Vec<Instruction>,
    symbol_table: &'a mut SymbolTable,
}

enum ExpResult {
    PlainOperand(Val),
    DereferencedPointer(Val),
}

impl<'a> InstructionGenerator<'a> {
    fn new(symbol_table: &'a mut SymbolTable) -> Self {
        Self {
            var_counter: 0,
            instructions: Vec::new(),
            symbol_table,
        }
    }

    fn make_tmp_local(&mut self, ty: ast::VarType) -> Val {
        let var = EcoString::from(format!("tmp.{}", self.var_counter));
        self.var_counter += 1;
        self.symbol_table.insert(var.clone(), Attr::Local(ty));
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
        if decl.storage_class.is_some() {
            return;
        }
        if let Some(exp) = &decl.init {
            let val = self.add_expression_and_convert(exp);
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
                let val = self.add_expression_and_convert(expression);
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
                    let cond = self.add_expression_and_convert(condition);
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
                    let cond = self.add_expression_and_convert(condition);
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

                let cond = self.add_expression_and_convert(condition);
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
                let cond = self.add_expression_and_convert(condition);
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
                    let cond = self.add_expression_and_convert(condition);
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

    fn add_expression(&mut self, expression: &ast::Expression) -> ExpResult {
        match expression {
            ast::Expression::Unary {
                op: Spanned { data: op, .. },
                exp,
                ty,
            } => {
                let src = self.add_expression_and_convert(exp);
                let dst = self.make_tmp_local(ty.clone());
                self.instructions.push(Instruction::Unary {
                    op: match op {
                        ast::UnaryOp::Negate => UnaryOp::Negate,
                        ast::UnaryOp::Complement => UnaryOp::Complement,
                        ast::UnaryOp::Not => UnaryOp::Not,
                    },
                    src,
                    dst: dst.clone(),
                });
                ExpResult::PlainOperand(dst)
            }
            ast::Expression::Binary {
                op: ast::BinaryOp::And,
                lhs,
                rhs,
                ty,
            } => {
                let lhs = self.add_expression_and_convert(lhs);
                let dst = self.make_tmp_local(ty.clone());
                let and_false = self.new_label("and_false");
                self.instructions.push(Instruction::JumpIfZero {
                    src: lhs.clone(),
                    dst: and_false.clone(),
                });
                let rhs = self.add_expression_and_convert(rhs);
                self.instructions.push(Instruction::JumpIfZero {
                    src: rhs.clone(),
                    dst: and_false.clone(),
                });
                self.instructions.push(Instruction::Copy {
                    src: Val::Constant(ast::Const::Int(1)),
                    dst: dst.clone(),
                });
                let end = self.new_label("and_end");
                self.instructions.push(Instruction::Jump(end.clone()));
                self.instructions
                    .push(Instruction::Label(and_false.clone()));
                self.instructions.push(Instruction::Copy {
                    src: Val::Constant(ast::Const::Int(0)),
                    dst: dst.clone(),
                });
                self.instructions.push(Instruction::Label(end));
                ExpResult::PlainOperand(dst)
            }
            ast::Expression::Binary {
                op: ast::BinaryOp::Or,
                lhs,
                rhs,
                ty,
            } => {
                let lhs = self.add_expression_and_convert(lhs);
                let dst = self.make_tmp_local(ty.clone());
                let or_true = self.new_label("or_true");
                self.instructions.push(Instruction::JumpIfNotZero {
                    src: lhs.clone(),
                    dst: or_true.clone(),
                });
                let rhs = self.add_expression_and_convert(rhs);
                self.instructions.push(Instruction::JumpIfNotZero {
                    src: rhs.clone(),
                    dst: or_true.clone(),
                });
                self.instructions.push(Instruction::Copy {
                    src: Val::Constant(ast::Const::Int(0)),
                    dst: dst.clone(),
                });
                let end = self.new_label("or_end");
                self.instructions.push(Instruction::Jump(end.clone()));
                self.instructions.push(Instruction::Label(or_true.clone()));
                self.instructions.push(Instruction::Copy {
                    src: Val::Constant(ast::Const::Int(1)),
                    dst: dst.clone(),
                });
                self.instructions.push(Instruction::Label(end));
                ExpResult::PlainOperand(dst)
            }
            ast::Expression::Binary { op, lhs, rhs, ty } => {
                let lhs = self.add_expression_and_convert(lhs);
                let rhs = self.add_expression_and_convert(rhs);
                let dst = self.make_tmp_local(ty.clone());
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
                        ast::BinaryOp::And | ast::BinaryOp::Or => unreachable!(),
                    },
                    lhs,
                    rhs,
                    dst: dst.clone(),
                });
                ExpResult::PlainOperand(dst)
            }
            ast::Expression::Var(Spanned { data: var, .. }, _) => {
                ExpResult::PlainOperand(Val::Var(var.clone()))
            }
            ast::Expression::Assignment { lhs, rhs } => {
                let lhs = self.add_expression(lhs);
                let rhs = self.add_expression_and_convert(rhs);

                match &lhs {
                    ExpResult::PlainOperand(dst) => {
                        self.instructions.push(Instruction::Copy {
                            src: rhs,
                            dst: dst.clone(),
                        });
                        lhs
                    }
                    ExpResult::DereferencedPointer(ptr) => {
                        self.instructions.push(Instruction::Store {
                            src: rhs.clone(),
                            dst: ptr.clone(),
                        });
                        ExpResult::PlainOperand(rhs)
                    }
                }
            }
            ast::Expression::Conditional {
                condition,
                then_branch,
                else_branch,
            } => {
                let dst = self.make_tmp_local(then_branch.ty().clone());
                let else_label = self.new_label("cond_else");
                let end_label = self.new_label("cond_end");
                let cond = self.add_expression_and_convert(condition);
                self.instructions.push(Instruction::JumpIfZero {
                    src: cond,
                    dst: else_label.clone(),
                });
                let v1 = self.add_expression_and_convert(then_branch);
                self.instructions.push(Instruction::Copy {
                    src: v1,
                    dst: dst.clone(),
                });
                self.instructions.push(Instruction::Jump(end_label.clone()));
                self.instructions.push(Instruction::Label(else_label));
                let v2 = self.add_expression_and_convert(else_branch);
                self.instructions.push(Instruction::Copy {
                    src: v2,
                    dst: dst.clone(),
                });
                self.instructions.push(Instruction::Label(end_label));
                ExpResult::PlainOperand(dst)
            }
            ast::Expression::FunctionCall { name, args, ty } => {
                let dst = self.make_tmp_local(ty.clone());
                let args = args
                    .iter()
                    .map(|arg| self.add_expression_and_convert(arg))
                    .collect::<Vec<_>>();
                self.instructions.push(Instruction::FunCall {
                    name: name.data.clone(),
                    args,
                    dst: dst.clone(),
                });
                ExpResult::PlainOperand(dst)
            }
            ast::Expression::Cast { target, exp } => {
                let val = self.add_expression_and_convert(exp);
                match (exp.ty(), target) {
                    (from, to) if from == to => ExpResult::PlainOperand(val),
                    (ast::VarType::Double, to @ (ast::VarType::Int | ast::VarType::Long)) => {
                        let dst = self.make_tmp_local(to.clone());
                        self.instructions.push(Instruction::DoubleToInt {
                            src: val,
                            dst: dst.clone(),
                        });
                        ExpResult::PlainOperand(dst)
                    }
                    (ast::VarType::Double, to @ (ast::VarType::Uint | ast::VarType::Ulong)) => {
                        let dst = self.make_tmp_local(to.clone());
                        self.instructions.push(Instruction::DoubleToUint {
                            src: val,
                            dst: dst.clone(),
                        });
                        ExpResult::PlainOperand(dst)
                    }
                    (ast::VarType::Int | ast::VarType::Long, ast::VarType::Double) => {
                        let dst = self.make_tmp_local(ast::VarType::Double);
                        self.instructions.push(Instruction::IntToDouble {
                            src: val,
                            dst: dst.clone(),
                        });
                        ExpResult::PlainOperand(dst)
                    }
                    (ast::VarType::Uint | ast::VarType::Ulong, ast::VarType::Double) => {
                        let dst = self.make_tmp_local(ast::VarType::Double);
                        self.instructions.push(Instruction::UintToDouble {
                            src: val,
                            dst: dst.clone(),
                        });
                        ExpResult::PlainOperand(dst)
                    }
                    (from, to) if from.size() == to.size() => {
                        let dst = self.make_tmp_local(to.clone());
                        self.instructions.push(Instruction::Copy {
                            src: val,
                            dst: dst.clone(),
                        });
                        ExpResult::PlainOperand(dst)
                    }
                    (from, to) if from.size() > to.size() => {
                        let dst = self.make_tmp_local(to.clone());
                        self.instructions.push(Instruction::Truncate {
                            src: val,
                            dst: dst.clone(),
                        });
                        ExpResult::PlainOperand(dst)
                    }
                    (from, to) if from.is_signed() => {
                        let dst = self.make_tmp_local(to.clone());
                        self.instructions.push(Instruction::SignExtend {
                            src: val,
                            dst: dst.clone(),
                        });
                        ExpResult::PlainOperand(dst)
                    }
                    (_, to) => {
                        let dst = self.make_tmp_local(to.clone());
                        self.instructions.push(Instruction::ZeroExtend {
                            src: val,
                            dst: dst.clone(),
                        });
                        ExpResult::PlainOperand(dst)
                    }
                }
            }
            ast::Expression::Constant(c) => ExpResult::PlainOperand(Val::Constant(c.data)),
            ast::Expression::Dereference(exp) => {
                let val = self.add_expression_and_convert(exp);
                ExpResult::DereferencedPointer(val)
            }
            ast::Expression::AddrOf { exp, ty } => {
                let val = self.add_expression(exp);
                match val {
                    ExpResult::PlainOperand(val) => {
                        let dst = self.make_tmp_local(ty.clone());
                        self.instructions.push(Instruction::GetAddress {
                            src: val,
                            dst: dst.clone(),
                        });
                        ExpResult::PlainOperand(dst)
                    }
                    ExpResult::DereferencedPointer(ptr) => ExpResult::PlainOperand(ptr),
                }
            }
        }
    }

    fn add_expression_and_convert(&mut self, expression: &ast::Expression) -> Val {
        match self.add_expression(expression) {
            ExpResult::PlainOperand(val) => val,
            ExpResult::DereferencedPointer(ptr) => {
                let dst = self.make_tmp_local(ptr.ty(self.symbol_table));
                self.instructions.push(Instruction::Load {
                    src: ptr,
                    dst: dst.clone(),
                });
                dst
            }
        }
    }
}

pub fn gen_program(program: &ast::Program, symbol_table: &mut HashMap<EcoString, Attr>) -> Program {
    let mut generator = InstructionGenerator::new(symbol_table);
    Program {
        top_levels: generator
            .symbol_table
            .iter()
            .filter_map(|(key, value)| {
                if let Attr::Static { init, global, ty } = value {
                    let init = match init {
                        semantics::type_check::InitialValue::Initial(i) => *i,
                        semantics::type_check::InitialValue::Tentative => ty.zero(),
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
            .collect::<Vec<_>>()
            .into_iter()
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
                data: if function.ty.ret == VarType::Double {
                    ast::Const::Double(0.0)
                } else {
                    ast::Const::Int(0)
                },
                span: 0..0,
            },
        )));
        Some(Function {
            global: if let Attr::Fun { global, .. } = generator.symbol_table[&function.name.data] {
                global
            } else {
                unreachable!()
            },
            name: function.name.data.clone(),
            params: function.params.iter().map(|s| s.data.clone()).collect(),
            body: std::mem::take(&mut generator.instructions),
        })
    } else {
        None
    }
}
