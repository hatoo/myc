use std::collections::HashMap;

use ecow::EcoString;

use crate::{
    ast::Expression,
    span::{HasSpan, Spanned},
};

#[derive(Debug, Default)]
pub struct TypeChecker {
    pub sym_table: HashMap<EcoString, Attr>,
}

#[derive(Debug)]
pub enum Attr {
    Fun {
        arity: usize,
        defined: bool,
        global: bool,
    },
    Static {
        init: InitialValue,
        global: bool,
    },
    Local,
}

#[derive(Debug, Clone, Copy)]
pub enum InitialValue {
    Tentative,
    Initial(i32),
    NoInitializer,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Incompatible types: {0}")]
    IncompatibleTypes(Spanned<EcoString>),
    #[error("Function redefined: {0}")]
    Redefined(Spanned<EcoString>),
    #[error("Static function declaration follows non-static : {0}")]
    StaticFunAfterNonStatic(Spanned<EcoString>),
    #[error("Bad Initializer")]
    BadInitializer(Spanned<EcoString>),
    #[error("Incompatible linkage: {0}")]
    IncompatibleLinkage(Spanned<EcoString>),
}

impl HasSpan for Error {
    fn span(&self) -> std::ops::Range<usize> {
        match self {
            Error::IncompatibleTypes(ident) => ident.span.clone(),
            Error::Redefined(ident) => ident.span.clone(),
            Error::StaticFunAfterNonStatic(ident) => ident.span.clone(),
            Error::BadInitializer(ident) => ident.span.clone(),
            Error::IncompatibleLinkage(ident) => ident.span.clone(),
        }
    }
}

impl TypeChecker {
    pub fn check_program(&mut self, program: &crate::ast::Program) -> Result<(), Error> {
        for decl in &program.decls {
            match decl {
                crate::ast::Declaration::VarDecl(decl) => self.check_var_decl_file(decl)?,
                crate::ast::Declaration::FunDecl(decl) => self.check_fun_decl(decl)?,
            }
        }

        Ok(())
    }

    fn check_fun_decl(&mut self, fun_decl: &crate::ast::FunDecl) -> Result<(), Error> {
        let crate::ast::FunDecl {
            name,
            params,
            body,
            storage_class,
        } = fun_decl;

        let mut new_global = storage_class != &Some(crate::ast::StorageClass::Static);
        let mut already_defined = false;

        if let Some(attr) = self.sym_table.get(&name.data) {
            if let Attr::Fun {
                arity,
                defined,
                global,
            } = attr
            {
                if *arity != params.len() {
                    return Err(Error::IncompatibleTypes(name.clone()));
                }
                if *defined && body.is_some() {
                    return Err(Error::Redefined(name.clone()));
                }
                already_defined = *defined;
                if *global && storage_class == &Some(crate::ast::StorageClass::Static) {
                    return Err(Error::StaticFunAfterNonStatic(name.clone()));
                }

                new_global = *global;
            } else {
                return Err(Error::IncompatibleTypes(name.clone()));
            }
        }

        self.sym_table.insert(
            name.data.clone(),
            Attr::Fun {
                arity: params.len(),
                defined: already_defined || body.is_some(),
                global: new_global,
            },
        );

        if let Some(body) = body {
            for param in params {
                self.sym_table.insert(param.data.clone(), Attr::Local);
            }
            self.check_block(body)?;
        }

        Ok(())
    }

    fn check_block(&mut self, block: &crate::ast::Block) -> Result<(), Error> {
        for block_item in &block.0 {
            match block_item {
                crate::ast::BlockItem::Declaration(decl) => self.check_decl(decl)?,
                crate::ast::BlockItem::Statement(stmt) => self.check_statement(stmt)?,
            }
        }
        Ok(())
    }

    fn check_decl(&mut self, decl: &crate::ast::Declaration) -> Result<(), Error> {
        match decl {
            crate::ast::Declaration::VarDecl(decl) => self.check_var_decl_local(decl),
            crate::ast::Declaration::FunDecl(decl) => {
                if decl.body.is_some() {
                    return Err(Error::IncompatibleTypes(decl.name.clone()));
                }
                self.check_fun_decl(decl)
            }
        }
    }

    fn check_var_decl_file(&mut self, decl: &crate::ast::VarDecl) -> Result<(), Error> {
        let crate::ast::VarDecl {
            ident,
            init,
            storage_class,
        } = decl;

        let mut init = match init {
            Some(Expression::Constant(val)) => InitialValue::Initial(val.data),
            None => {
                if storage_class == &Some(crate::ast::StorageClass::Extern) {
                    InitialValue::NoInitializer
                } else {
                    InitialValue::Tentative
                }
            }
            _ => return Err(Error::BadInitializer(ident.clone())),
        };

        let mut global = storage_class != &Some(crate::ast::StorageClass::Static);

        match self.sym_table.get(&ident.data) {
            Some(Attr::Fun { .. }) => {
                return Err(Error::IncompatibleTypes(ident.clone()));
            }
            Some(Attr::Static {
                init: old_init,
                global: old_global,
            }) => {
                if storage_class == &Some(crate::ast::StorageClass::Extern) {
                    global = *old_global;
                } else if *old_global != global {
                    return Err(Error::IncompatibleLinkage(ident.clone()));
                }

                if matches!(old_init, InitialValue::Initial(_)) {
                    if matches!(init, InitialValue::Initial(_)) {
                        return Err(Error::BadInitializer(ident.clone()));
                    }
                    init = *old_init;
                } else if !matches!(init, InitialValue::Initial(_))
                    && matches!(old_init, InitialValue::Tentative)
                {
                    init = InitialValue::Tentative;
                }
            }
            Some(Attr::Local) => {
                unreachable!()
            }
            None => {}
        }

        self.sym_table
            .insert(ident.data.clone(), Attr::Static { init, global });
        Ok(())
    }

    fn check_var_decl_local(&mut self, decl: &crate::ast::VarDecl) -> Result<(), Error> {
        let crate::ast::VarDecl {
            ident,
            init,
            storage_class,
        } = decl;

        match storage_class {
            Some(crate::ast::StorageClass::Extern) => {
                if init.is_some() {
                    return Err(Error::BadInitializer(ident.clone()));
                }
                match self.sym_table.get(&ident.data) {
                    Some(Attr::Fun { .. }) => {
                        return Err(Error::IncompatibleTypes(ident.clone()));
                    }
                    Some(_) => {}
                    None => {
                        self.sym_table.insert(
                            ident.data.clone(),
                            Attr::Static {
                                init: InitialValue::NoInitializer,
                                global: true,
                            },
                        );
                    }
                }
            }
            Some(crate::ast::StorageClass::Static) => {
                let init = match init {
                    Some(Expression::Constant(val)) => InitialValue::Initial(val.data),
                    None => InitialValue::Initial(0),
                    _ => return Err(Error::BadInitializer(ident.clone())),
                };
                self.sym_table.insert(
                    ident.data.clone(),
                    Attr::Static {
                        init,
                        global: false,
                    },
                );
            }
            _ => {
                self.sym_table.insert(ident.data.clone(), Attr::Local);
                if let Some(exp) = init {
                    self.check_expression(exp)?;
                }
            }
        }

        Ok(())
    }

    fn check_expression(&mut self, exp: &crate::ast::Expression) -> Result<(), Error> {
        match exp {
            crate::ast::Expression::Var(name) => {
                if let Some(Attr::Fun { .. }) = self.sym_table.get(&name.data) {
                    Err(Error::IncompatibleTypes(name.clone()))
                } else {
                    Ok(())
                }
            }
            crate::ast::Expression::Constant(_) => Ok(()),
            crate::ast::Expression::Unary { op: _, exp } => self.check_expression(exp),
            crate::ast::Expression::Binary { op: _, lhs, rhs } => {
                self.check_expression(lhs)?;
                self.check_expression(rhs)?;
                Ok(())
            }
            crate::ast::Expression::Assignment { lhs, rhs } => {
                self.check_expression(lhs)?;
                self.check_expression(rhs)?;
                Ok(())
            }
            crate::ast::Expression::Conditional {
                condition,
                then_branch,
                else_branch,
            } => {
                self.check_expression(condition)?;
                self.check_expression(then_branch)?;
                self.check_expression(else_branch)?;
                Ok(())
            }
            crate::ast::Expression::FunctionCall { name, args } => {
                if let Some(Attr::Fun { arity, .. }) = self.sym_table.get(&name.data) {
                    if *arity != args.len() {
                        return Err(Error::IncompatibleTypes(name.clone()));
                    }
                    for arg in args {
                        self.check_expression(arg)?;
                    }
                    Ok(())
                } else {
                    Err(Error::IncompatibleTypes(name.clone()))
                }
            }
        }
    }

    fn check_statement(&mut self, stmt: &crate::ast::Statement) -> Result<(), Error> {
        match stmt {
            crate::ast::Statement::Return(exp) => self.check_expression(exp),
            crate::ast::Statement::Expression(exp) => self.check_expression(exp),
            crate::ast::Statement::If {
                condition,
                then_branch,
                else_branch,
            } => {
                self.check_expression(condition)?;
                self.check_statement(then_branch)?;
                if let Some(else_branch) = else_branch {
                    self.check_statement(else_branch)?;
                }
                Ok(())
            }
            crate::ast::Statement::Compound(block) => {
                self.check_block(block)?;
                Ok(())
            }
            crate::ast::Statement::Break { .. } => Ok(()),
            crate::ast::Statement::Continue { .. } => Ok(()),
            crate::ast::Statement::While {
                label: _,
                condition,
                body,
            } => {
                self.check_expression(condition)?;
                self.check_statement(body)?;
                Ok(())
            }
            crate::ast::Statement::DoWhile {
                label: _,
                condition,
                body,
            } => {
                self.check_statement(body)?;
                self.check_expression(condition)?;
                Ok(())
            }
            crate::ast::Statement::For {
                label: _,
                init,
                condition,
                step,
                body,
            } => {
                if let Some(init) = init {
                    match init {
                        crate::ast::ForInit::VarDecl(decl) => self.check_var_decl_local(decl)?,
                        crate::ast::ForInit::Expression(exp) => self.check_expression(exp)?,
                    }
                }
                if let Some(condition) = condition {
                    self.check_expression(condition)?;
                }
                if let Some(step) = step {
                    self.check_expression(step)?;
                }
                self.check_statement(body)?;
                Ok(())
            }
            crate::ast::Statement::Null => Ok(()),
        }
    }
}
