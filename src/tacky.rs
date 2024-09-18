use ecow::EcoString;

use crate::{
    ast::{self, BaseType, Block, Const, Expression, Initializer, Ty, VarType},
    lexer::TokenSpanned,
    semantics::{
        self,
        type_check::{Attr, StaticInit, SymbolTable},
    },
};

#[derive(Debug)]
pub struct Program {
    pub top_levels: Vec<TopLevelItem>,
}

#[derive(Debug)]
pub enum TopLevelItem {
    Function(Function),
    StaticVariable(StaticVariable),
    StaticConstant(StaticConstant),
}

#[derive(Debug)]
pub struct Function {
    pub global: bool,
    pub name: EcoString,
    pub params: Vec<EcoString>,
    pub body: Vec<Instruction>,
    pub return_ty: ast::VarType,
}

#[derive(Debug)]
pub struct StaticVariable {
    pub global: bool,
    pub name: EcoString,
    pub alignment: usize,
    pub init: Vec<semantics::type_check::StaticInit>,
}

#[derive(Debug)]
pub struct StaticConstant {
    pub name: EcoString,
    pub ty: ast::VarType,
    pub init: semantics::type_check::StaticInit,
}

#[derive(Debug, Clone, PartialEq)]
// We use `Val` as operand even if it's not gonna be a constant (e.g. dst) for easiness.
pub enum Instruction {
    Nop,
    Return(Option<Val>),
    Cast {
        src: Val,
        dst: Val,
    },
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
        callee: Val,
        args: Vec<Val>,
        dst: Option<Val>,
    },
    AddPtr {
        ptr: Val,
        index: Val,
        scale: usize,
        dst: Val,
    },
    CopyToOffset {
        src: Val,
        dst: Val,
        offset: usize,
    },
    CopyFromOffset {
        src: Val,
        offset: usize,
        dst: Val,
    },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnaryOp {
    Negate,
    Complement,
    Not,
}

#[derive(Debug, Clone, PartialEq, Eq)]
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
    BitAnd,
    BitOr,
    Xor,
    ShiftLeft,
    ShiftRight,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum Val {
    Constant(ast::Const),
    Var(EcoString),
}

impl Val {
    pub fn var(&self) -> &EcoString {
        match self {
            Val::Constant(_) => panic!("Expected variable, found constant"),
            Val::Var(var) => var,
        }
    }

    pub fn ty(&self, symbol_table: &SymbolTable) -> ast::VarType {
        match self {
            Val::Constant(c) => match c {
                ast::Const::Int(_) => ast::BaseType::Int.into(),
                ast::Const::Long(_) => ast::BaseType::Long.into(),
                ast::Const::Uint(_) => ast::BaseType::Uint.into(),
                ast::Const::Ulong(_) => ast::BaseType::Ulong.into(),
                ast::Const::Double(_) => ast::BaseType::Double.into(),
                ast::Const::Char(_) => ast::BaseType::Char.into(),
                ast::Const::UChar(_) => ast::BaseType::UChar.into(),
            },
            Val::Var(var) => match &symbol_table[var] {
                Attr::Fun { ty, .. } => ty.ret.clone(),
                Attr::Static { ty, .. } => ty.clone(),
                Attr::Local(ty) => ty.clone(),
                Attr::Constant { ty, .. } => ty.clone(),
                Attr::Struct(_) => ast::VarType::Struct(var.clone()),
                Attr::Union(_) => ast::VarType::Union(var.clone()),
            },
        }
    }

    pub fn is_static(&self, symbol_table: &SymbolTable) -> bool {
        match self {
            Val::Constant(_) => false,
            Val::Var(var) => matches!(symbol_table[var], Attr::Static { .. }),
        }
    }

    pub fn is_constant(&self) -> bool {
        matches!(self, Val::Constant(_))
    }
}

struct InstructionGenerator<'a> {
    var_counter: usize,
    instructions: Vec<Instruction>,
    symbol_table: &'a mut SymbolTable,
}

#[derive(Debug)]
enum ExpResult {
    PlainOperand(Val),
    DereferencedPointer(Val),
    SubObject { base: EcoString, offset: usize },
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
                ast::Declaration::StructDecl(_) => {}
                ast::Declaration::UnionDecl(_) => {}
            },
            ast::BlockItem::Statement(stmt) => {
                self.add_statement(stmt);
            }
        }
    }

    fn copy_initializers(
        &mut self,
        init: &Initializer,
        name: EcoString,
        target: &VarType,
        offset: &mut usize,
        depth: usize,
    ) {
        match init {
            Initializer::SingleInit(Expression::String(data, ty @ VarType::Array { .. })) => {
                for chunk in data
                    .data
                    .iter()
                    .chain(std::iter::repeat(&0))
                    .take(self.symbol_table.size(ty))
                    .copied()
                    .collect::<Vec<_>>()
                    .chunks(4)
                {
                    if chunk.len() == 4 {
                        let val = Val::Constant(ast::Const::Uint(u32::from_le_bytes([
                            chunk[0], chunk[1], chunk[2], chunk[3],
                        ])));
                        self.instructions.push(Instruction::CopyToOffset {
                            src: val,
                            dst: Val::Var(name.clone()),
                            offset: *offset,
                        });
                        *offset += chunk.len();
                    } else {
                        for byte in chunk {
                            let val = Val::Constant(ast::Const::UChar(*byte));
                            self.instructions.push(Instruction::CopyToOffset {
                                src: val,
                                dst: Val::Var(name.clone()),
                                offset: *offset,
                            });
                            *offset += 1;
                        }
                    }
                }
            }
            Initializer::SingleInit(exp) => {
                let val = self.add_expression_and_convert(exp);
                let size = self.symbol_table.size(&val.ty(self.symbol_table));
                if depth == 0 {
                    self.instructions.push(Instruction::Copy {
                        src: val.clone(),
                        dst: Val::Var(name.clone()),
                    });
                } else {
                    self.instructions.push(Instruction::CopyToOffset {
                        src: val,
                        dst: Val::Var(name.clone()),
                        offset: *offset,
                    });
                }
                *offset += size;
            }
            Initializer::CompoundInit(inits) => match target {
                VarType::Array { element, .. } => {
                    for init in inits {
                        self.copy_initializers(init, name.clone(), element, offset, depth + 1);
                    }
                }
                VarType::Struct(struct_name) => {
                    let struct_def = self.symbol_table.struct_def(struct_name).clone();
                    let offset_start = *offset;
                    for (member, init) in struct_def.members.iter().zip(inits) {
                        self.copy_initializers(
                            init,
                            name.clone(),
                            &member.ty,
                            &mut (offset_start + member.offset),
                            depth + 1,
                        );
                    }
                    *offset = offset_start + struct_def.size;
                }
                _ => unreachable!(),
            },
        }
    }

    fn add_var_declaration(&mut self, decl: &ast::VarDecl) {
        if decl.storage_class.is_some() {
            return;
        }

        if let Some(init) = decl.init.as_ref() {
            self.copy_initializers(init, decl.ident.data.clone(), &decl.ty, &mut 0, 0);
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
                if let Some(exp) = expression {
                    let val = self.add_expression_and_convert(exp);
                    self.instructions.push(Instruction::Return(Some(val)));
                } else {
                    self.instructions.push(Instruction::Return(None));
                }
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
            ast::Statement::Goto(label) => {
                self.instructions
                    .push(Instruction::Jump(label.data.clone()));
            }
            ast::Statement::Label { label, statement } => {
                self.instructions
                    .push(Instruction::Label(label.data.clone()));
                self.add_statement(statement);
            }
            ast::Statement::Case {
                statement, label, ..
            } => {
                self.instructions.push(Instruction::Label(label.clone()));
                self.add_statement(statement);
            }
            ast::Statement::Default {
                statement, label, ..
            } => {
                self.instructions.push(Instruction::Label(label.clone()));
                self.add_statement(statement);
            }
            ast::Statement::Switch {
                exp,
                statement,
                label,
                labels,
            } => {
                let exp = self.add_expression_and_convert(exp);

                let break_label: EcoString = format!("break_{}", label).into();
                for (case, label) in labels.cases.iter() {
                    let cond = self.make_tmp_local(VarType::Base(BaseType::Int));

                    let case = match exp.ty(self.symbol_table) {
                        VarType::Base(base) => match base {
                            BaseType::Char | BaseType::SChar => ast::Const::Char(*case as _),
                            BaseType::UChar => ast::Const::UChar(*case as _),
                            BaseType::Int => ast::Const::Int(*case as _),
                            BaseType::Uint => ast::Const::Uint(*case as _),
                            BaseType::Long => ast::Const::Long(*case as _),
                            BaseType::Ulong => ast::Const::Ulong(*case as _),
                            BaseType::Double => unreachable!(),
                        },
                        _ => unreachable!(),
                    };

                    self.instructions.push(Instruction::Binary {
                        op: BinaryOp::Equal,
                        lhs: exp.clone(),
                        rhs: Val::Constant(case),
                        dst: cond.clone(),
                    });
                    self.instructions.push(Instruction::JumpIfNotZero {
                        src: cond,
                        dst: label.clone(),
                    });
                }
                if let Some(default) = &labels.default {
                    self.instructions.push(Instruction::Jump(default.clone()));
                } else {
                    self.instructions
                        .push(Instruction::Jump(break_label.clone()));
                }
                self.add_statement(statement);
                self.instructions.push(Instruction::Label(break_label));
            }
        }
    }

    fn add_expression(&mut self, expression: &ast::Expression) -> ExpResult {
        match expression {
            ast::Expression::Unary {
                op: TokenSpanned { data: op, .. },
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

                if let ast::VarType::Pointer(elem) = ty {
                    let ast::Ty::Var(elem) = elem.as_ref() else {
                        unreachable!()
                    };
                    match op {
                        ast::BinaryOp::Add => {
                            let (lhs, rhs) = if lhs.ty(self.symbol_table).is_pointer() {
                                (lhs, rhs)
                            } else {
                                (rhs, lhs)
                            };
                            self.instructions.push(Instruction::AddPtr {
                                ptr: lhs,
                                index: rhs,
                                scale: self.symbol_table.size(elem),
                                dst: dst.clone(),
                            });
                            return ExpResult::PlainOperand(dst);
                        }
                        ast::BinaryOp::Subtract => {
                            // ptr - int

                            let neg = self.make_tmp_local(ast::BaseType::Long.into());
                            self.instructions.push(Instruction::Unary {
                                op: UnaryOp::Negate,
                                src: rhs.clone(),
                                dst: neg.clone(),
                            });
                            self.instructions.push(Instruction::AddPtr {
                                ptr: lhs,
                                index: neg,
                                scale: self.symbol_table.size(elem),
                                dst: dst.clone(),
                            });
                            return ExpResult::PlainOperand(dst);
                        }
                        _ => unreachable!(),
                    }
                }

                if matches!(op, ast::BinaryOp::Subtract)
                    && lhs.ty(self.symbol_table).is_pointer()
                    && rhs.ty(self.symbol_table).is_pointer()
                {
                    let VarType::Pointer(elem) = lhs.ty(self.symbol_table) else {
                        unreachable!()
                    };
                    let elem_size = self.symbol_table.ty_size(&elem);
                    // ptr - ptr
                    let diff = self.make_tmp_local(ast::BaseType::Long.into());
                    self.instructions.push(Instruction::Binary {
                        op: BinaryOp::Subtract,
                        lhs: lhs.clone(),
                        rhs: rhs.clone(),
                        dst: diff.clone(),
                    });
                    self.instructions.push(Instruction::Binary {
                        op: BinaryOp::Divide,
                        lhs: diff.clone(),
                        rhs: Val::Constant(ast::Const::Long(elem_size as _)),
                        dst: dst.clone(),
                    });
                    return ExpResult::PlainOperand(dst);
                }

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
                        ast::BinaryOp::BitAnd => BinaryOp::BitAnd,
                        ast::BinaryOp::BitOr => BinaryOp::BitOr,
                        ast::BinaryOp::Xor => BinaryOp::Xor,
                        ast::BinaryOp::ShiftLeft => BinaryOp::ShiftLeft,
                        ast::BinaryOp::ShiftRight => BinaryOp::ShiftRight,
                    },
                    lhs,
                    rhs,
                    dst: dst.clone(),
                });
                ExpResult::PlainOperand(dst)
            }
            ast::Expression::Var(TokenSpanned { data: var, .. }, _) => {
                if let semantics::type_check::Attr::Fun { ty, .. } = &self.symbol_table[var] {
                    let tmp = self
                        .make_tmp_local(ast::VarType::Pointer(Box::new(ast::Ty::Fun(ty.clone()))));
                    self.instructions.push(Instruction::GetAddress {
                        src: Val::Var(var.clone()),
                        dst: tmp.clone(),
                    });
                    ExpResult::PlainOperand(tmp)
                } else {
                    ExpResult::PlainOperand(Val::Var(var.clone()))
                }
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
                    ExpResult::SubObject { base, offset } => {
                        self.instructions.push(Instruction::CopyToOffset {
                            src: rhs.clone(),
                            dst: Val::Var(base.clone()),
                            offset: *offset,
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
                if then_branch.ty() != &ast::VarType::Void {
                    self.instructions.push(Instruction::Copy {
                        src: v1,
                        dst: dst.clone(),
                    });
                }
                self.instructions.push(Instruction::Jump(end_label.clone()));
                self.instructions.push(Instruction::Label(else_label));
                let v2 = self.add_expression_and_convert(else_branch);
                if else_branch.ty() != &ast::VarType::Void {
                    self.instructions.push(Instruction::Copy {
                        src: v2,
                        dst: dst.clone(),
                    });
                }
                self.instructions.push(Instruction::Label(end_label));
                ExpResult::PlainOperand(dst)
            }
            ast::Expression::FunctionCall { callee, args, ty } => {
                let dst = if ty == &ast::VarType::Void {
                    None
                } else {
                    Some(self.make_tmp_local(ty.clone()))
                };

                let callee = if let ast::Expression::Var(name, _) = callee.as_ref() {
                    if let Attr::Fun { .. } = self.symbol_table[&name.data] {
                        Val::Var(name.data.clone())
                    } else {
                        self.add_expression_and_convert(callee)
                    }
                } else {
                    self.add_expression_and_convert(callee)
                };

                self.make_tmp_local(ty.clone());
                let args = args
                    .iter()
                    .map(|arg| self.add_expression_and_convert(arg))
                    .collect::<Vec<_>>();
                self.instructions.push(Instruction::FunCall {
                    callee,
                    args,
                    dst: dst.clone(),
                });
                ExpResult::PlainOperand(dst.unwrap_or(Val::Var("DUMMY_VAR".into())))
            }
            ast::Expression::Cast { target, exp } => {
                let val = self.add_expression_and_convert(exp);
                if target == &ast::VarType::Void {
                    return ExpResult::PlainOperand(Val::Var("DUMMY_VAR".into()));
                }
                if exp.ty() == target
                    || matches!(
                        (exp.ty(), target),
                        (
                            VarType::Base(BaseType::Char | BaseType::SChar),
                            VarType::Base(BaseType::Char | BaseType::SChar)
                        )
                    )
                {
                    ExpResult::PlainOperand(val)
                } else {
                    let dst = self.make_tmp_local(target.clone());
                    self.instructions.push(Instruction::Cast {
                        src: val,
                        dst: dst.clone(),
                    });
                    ExpResult::PlainOperand(dst)
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
                    ExpResult::SubObject { base, offset } => {
                        let dst = self.make_tmp_local(ty.clone());
                        let ptr = self.make_tmp_local(ast::VarType::Pointer(Box::new(
                            ast::Ty::Var(ty.clone()),
                        )));

                        self.instructions.push(Instruction::GetAddress {
                            src: Val::Var(base),
                            dst: ptr.clone(),
                        });
                        self.instructions.push(Instruction::AddPtr {
                            ptr,
                            index: Val::Constant(ast::Const::Int(offset as _)),
                            scale: 1,
                            dst: dst.clone(),
                        });
                        ExpResult::PlainOperand(dst)
                    }
                }
            }
            ast::Expression::Subscript { array, index, ty } => {
                let lhs = self.add_expression_and_convert(array);
                let rhs = self.add_expression_and_convert(index);

                let (ptr, index) = if lhs.ty(self.symbol_table).is_pointer() {
                    (lhs, rhs)
                } else {
                    (rhs, lhs)
                };

                let dst = self.make_tmp_local(ptr.ty(self.symbol_table).clone());

                self.instructions.push(Instruction::AddPtr {
                    ptr,
                    index,
                    scale: self.symbol_table.size(ty),
                    dst: dst.clone(),
                });

                ExpResult::DereferencedPointer(dst)
            }
            ast::Expression::String(data, ty) => {
                let name = self.new_label("tacky.string");
                let pad = self.symbol_table.size(ty) - data.data.len();
                self.symbol_table.insert(
                    name.clone(),
                    semantics::type_check::Attr::Constant {
                        ty: ty.clone(),
                        init: semantics::type_check::StaticInit::String {
                            data: data.data.clone(),
                            pad,
                        },
                    },
                );

                ExpResult::PlainOperand(Val::Var(name))
            }
            ast::Expression::Sizeof(exp) => {
                let size = self.symbol_table.size(exp.ty());
                ExpResult::PlainOperand(Val::Constant(ast::Const::Ulong(size as _)))
            }
            ast::Expression::SizeofType(ty) => {
                let size = self.symbol_table.size(&ty.data);
                ExpResult::PlainOperand(Val::Constant(ast::Const::Ulong(size as _)))
            }
            ast::Expression::Dot {
                structure,
                member,
                ty,
            } => match structure.ty() {
                ast::VarType::Struct(struct_name) => {
                    let struct_def = self.symbol_table.struct_def(struct_name);
                    let member_offset = struct_def
                        .members
                        .iter()
                        .find(|m| m.name == member.data)
                        .unwrap()
                        .offset;

                    match self.add_expression(structure) {
                        ExpResult::PlainOperand(Val::Var(v)) => ExpResult::SubObject {
                            base: v,
                            offset: member_offset,
                        },
                        ExpResult::PlainOperand(Val::Constant(_)) => unreachable!(),
                        ExpResult::SubObject { base, offset } => ExpResult::SubObject {
                            base,
                            offset: offset + member_offset,
                        },
                        ExpResult::DereferencedPointer(ptr) => {
                            let dst_ptr = self.make_tmp_local(ast::VarType::Pointer(Box::new(
                                ast::Ty::Var(ty.clone()),
                            )));

                            self.instructions.push(Instruction::AddPtr {
                                ptr,
                                index: Val::Constant(ast::Const::Int(member_offset as _)),
                                scale: 1,
                                dst: dst_ptr.clone(),
                            });

                            ExpResult::DereferencedPointer(dst_ptr)
                        }
                    }
                }
                ast::VarType::Union(_) => self.add_expression(structure),
                _ => unreachable!(),
            },
            ast::Expression::Arrow {
                pointer,
                member,
                ty,
            } => {
                let pointer_ty = if let ast::VarType::Pointer(ty) = pointer.ty() {
                    let ast::Ty::Var(t) = ty.as_ref() else {
                        unreachable!()
                    };
                    t
                } else {
                    unreachable!()
                };

                match pointer_ty {
                    ast::VarType::Struct(struct_name) => {
                        let struct_def = self.symbol_table.struct_def(struct_name);
                        let member_offset = struct_def
                            .members
                            .iter()
                            .find(|m| m.name == member.data)
                            .unwrap()
                            .offset;

                        let ptr = self.add_expression_and_convert(pointer);
                        let dst_ptr =
                            self.make_tmp_local(VarType::Pointer(Box::new(Ty::Var(ty.clone()))));

                        self.instructions.push(Instruction::AddPtr {
                            ptr,
                            index: Val::Constant(ast::Const::Int(member_offset as _)),
                            scale: 1,
                            dst: dst_ptr.clone(),
                        });

                        ExpResult::DereferencedPointer(dst_ptr)
                    }
                    ast::VarType::Union(_) => self.add_expression(pointer),
                    _ => unreachable!(),
                }
            }
            Expression::Increment { exp, postfix } => {
                let val = self.add_expression_and_convert(exp);
                let one = one_value(exp.ty(), self.symbol_table);

                if *postfix {
                    let dst = self.make_tmp_local(val.ty(self.symbol_table).clone());
                    self.instructions.push(Instruction::Copy {
                        src: val.clone(),
                        dst: dst.clone(),
                    });
                    self.instructions.push(Instruction::Binary {
                        op: BinaryOp::Add,
                        lhs: val.clone(),
                        rhs: Val::Constant(one),
                        dst: val.clone(),
                    });
                    ExpResult::PlainOperand(dst)
                } else {
                    self.instructions.push(Instruction::Binary {
                        op: BinaryOp::Add,
                        lhs: val.clone(),
                        rhs: Val::Constant(one),
                        dst: val.clone(),
                    });
                    ExpResult::PlainOperand(val)
                }
            }
            Expression::Decrement { exp, postfix } => {
                let val = self.add_expression_and_convert(exp);
                let one = one_value(exp.ty(), self.symbol_table);

                if *postfix {
                    let dst = self.make_tmp_local(val.ty(self.symbol_table).clone());
                    self.instructions.push(Instruction::Copy {
                        src: val.clone(),
                        dst: dst.clone(),
                    });
                    self.instructions.push(Instruction::Binary {
                        op: BinaryOp::Subtract,
                        lhs: val.clone(),
                        rhs: Val::Constant(one),
                        dst: val.clone(),
                    });
                    ExpResult::PlainOperand(dst)
                } else {
                    self.instructions.push(Instruction::Binary {
                        op: BinaryOp::Subtract,
                        lhs: val.clone(),
                        rhs: Val::Constant(one),
                        dst: val.clone(),
                    });
                    ExpResult::PlainOperand(val)
                }
            }
        }
    }

    fn add_expression_and_convert(&mut self, expression: &ast::Expression) -> Val {
        match self.add_expression(expression) {
            ExpResult::PlainOperand(val) => val,
            ExpResult::DereferencedPointer(ptr) => {
                let dst = self.make_tmp_local(expression.ty().clone());
                self.instructions.push(Instruction::Load {
                    src: ptr,
                    dst: dst.clone(),
                });
                dst
            }
            ExpResult::SubObject { base, offset } => {
                let dst = self.make_tmp_local(expression.ty().clone());
                self.instructions.push(Instruction::CopyFromOffset {
                    src: Val::Var(base),
                    offset,
                    dst: dst.clone(),
                });
                dst
            }
        }
    }
}

fn one_value(ty: &VarType, symbol_table: &SymbolTable) -> Const {
    match ty {
        VarType::Base(base) => match base {
            BaseType::Char | BaseType::SChar => ast::Const::Char(1),
            BaseType::UChar => ast::Const::UChar(1),
            BaseType::Int => ast::Const::Int(1),
            BaseType::Uint => ast::Const::Uint(1),
            BaseType::Long => ast::Const::Long(1),
            BaseType::Ulong => ast::Const::Ulong(1),
            BaseType::Double => ast::Const::Double(1.0),
        },
        VarType::Pointer(ty) => {
            let size = match ty.as_ref() {
                Ty::Var(ty) => symbol_table.size(ty),
                Ty::Fun(_) => 1,
            };
            ast::Const::Ulong(size as _)
        }
        _ => unreachable!(),
    }
}

pub fn gen_program(program: &ast::Program, symbol_table: &mut SymbolTable) -> Program {
    let mut generator = InstructionGenerator::new(symbol_table);

    let functions: Vec<_> = program
        .decls
        .iter()
        .filter_map(|f| match f {
            ast::Declaration::FunDecl(f) => gen_function(&mut generator, f),
            _ => None,
        })
        .map(TopLevelItem::Function)
        .collect();
    Program {
        top_levels: generator
            .symbol_table
            .iter()
            .filter_map(|(key, value)| match value {
                Attr::Static { init, global, ty } => {
                    let init = match init {
                        semantics::type_check::InitialValue::Initial(i) => i.clone(),
                        semantics::type_check::InitialValue::Tentative => {
                            vec![StaticInit::Zero(generator.symbol_table.size(ty))]
                        }
                        semantics::type_check::InitialValue::NoInitializer => return None,
                    };
                    Some(TopLevelItem::StaticVariable(StaticVariable {
                        global: *global,
                        name: key.clone(),
                        alignment: generator.symbol_table.alignment(ty),
                        init,
                    }))
                }
                Attr::Constant { ty, init } => Some(TopLevelItem::StaticConstant(StaticConstant {
                    name: key.clone(),
                    ty: ty.clone(),
                    init: init.clone(),
                })),
                _ => None,
            })
            .chain(functions)
            .collect(),
    }
}

fn gen_function(generator: &mut InstructionGenerator, function: &ast::FunDecl) -> Option<Function> {
    if let Some(block) = &function.body {
        for block_item in &block.0 {
            generator.add_block_item(block_item);
        }
        generator.add_statement(&ast::Statement::Return(match function.ty.ret {
            ast::VarType::Void => None,
            ast::VarType::Base(BaseType::Double) => Some(ast::Expression::Constant(TokenSpanned {
                data: ast::Const::Double(0.0),
                span: 0..0,
            })),
            _ => Some(ast::Expression::Constant(TokenSpanned {
                data: ast::Const::Int(0),
                span: 0..0,
            })),
        }));
        Some(Function {
            global: if let Attr::Fun { global, .. } = generator.symbol_table[&function.name.data] {
                global
            } else {
                unreachable!()
            },
            name: function.name.data.clone(),
            params: function.params.iter().map(|s| s.data.clone()).collect(),
            body: std::mem::take(&mut generator.instructions),
            return_ty: function.ty.ret.clone(),
        })
    } else {
        None
    }
}
