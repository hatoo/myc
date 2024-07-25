pub use loop_label::LoopLabel;
pub use var_resolve::VarResolver;

pub mod var_resolve {
    use std::collections::HashMap;

    use ecow::EcoString;

    use crate::{
        ast::{self, Expression},
        span::{HasSpan, Spanned},
    };
    #[derive(Debug, Default)]
    pub struct VarResolver {
        var_counter: usize,
        scopes: Vec<HashMap<EcoString, EcoString>>,
    }

    #[derive(Debug, thiserror::Error)]
    pub enum Error {
        #[error("Variable not declared: {0}")]
        VariableNotDeclared(Spanned<EcoString>),
        #[error("Variable already declared: {0}")]
        VariableAlreadyDeclared(Spanned<EcoString>),
        #[error("Invalid lvalue: {0:?}")]
        InvalidLValue(Expression),
    }

    impl HasSpan for Error {
        fn span(&self) -> std::ops::Range<usize> {
            match self {
                Error::VariableNotDeclared(ident) => ident.span.clone(),
                Error::VariableAlreadyDeclared(ident) => ident.span.clone(),
                Error::InvalidLValue(exp) => exp.span(),
            }
        }
    }

    impl VarResolver {
        pub fn new_var(&mut self, prefix: &EcoString) -> EcoString {
            let var = EcoString::from(format!("{}.{}", prefix, self.var_counter));
            self.var_counter += 1;
            var
        }

        pub fn lookup(&self, ident: &EcoString) -> Option<EcoString> {
            for scope in self.scopes.iter().rev() {
                if let Some(unique_name) = scope.get(ident) {
                    return Some(unique_name.clone());
                }
            }
            None
        }

        pub fn current_scope(&mut self) -> &mut HashMap<EcoString, EcoString> {
            self.scopes.last_mut().unwrap()
        }

        pub fn resolve_program(&mut self, program: &mut ast::Program) -> Result<(), Error> {
            self.scopes.push(HashMap::new());
            for block_item in &mut program.function_definition.body.0 {
                self.resolve_block_item(block_item)?;
            }
            self.scopes.pop().unwrap();
            Ok(())
        }

        pub fn resolve_block_item(&mut self, block_item: &mut ast::BlockItem) -> Result<(), Error> {
            match block_item {
                ast::BlockItem::Declaration(decl) => self.resolve_declaration(decl),
                ast::BlockItem::Statement(stmt) => self.resolve_statement(stmt),
            }
        }

        pub fn resolve_statement(&mut self, stmt: &mut ast::Statement) -> Result<(), Error> {
            match stmt {
                ast::Statement::Return(decl) => self.resolve_expression(decl),
                ast::Statement::Expression(exp) => self.resolve_expression(exp),
                ast::Statement::If {
                    condition,
                    then_branch,
                    else_branch,
                } => {
                    self.resolve_expression(condition)?;
                    self.resolve_statement(then_branch)?;
                    if let Some(else_branch) = else_branch {
                        self.resolve_statement(else_branch)?;
                    }
                    Ok(())
                }
                ast::Statement::Null => Ok(()),
                ast::Statement::Compound(stmts) => {
                    self.scopes.push(HashMap::new());
                    for block_item in &mut stmts.0 {
                        self.resolve_block_item(block_item)?;
                    }
                    self.scopes.pop().unwrap();
                    Ok(())
                }
                ast::Statement::Break { .. } => Ok(()),
                ast::Statement::Continue { .. } => Ok(()),
                ast::Statement::While {
                    condition, body, ..
                } => {
                    self.resolve_expression(condition)?;
                    self.resolve_statement(body)?;
                    Ok(())
                }
                ast::Statement::DoWhile {
                    condition, body, ..
                } => {
                    self.resolve_expression(condition)?;
                    self.resolve_statement(body)?;
                    Ok(())
                }
                ast::Statement::For {
                    init,
                    condition,
                    step,
                    body,
                    ..
                } => {
                    self.scopes.push(HashMap::new());
                    if let Some(for_init) = init {
                        match for_init {
                            ast::ForInit::Declaration(decl) => {
                                self.resolve_declaration(decl)?;
                            }
                            ast::ForInit::Expression(exp) => {
                                self.resolve_expression(exp)?;
                            }
                        }
                    }
                    if let Some(condition) = condition {
                        self.resolve_expression(condition)?;
                    }
                    if let Some(step) = step {
                        self.resolve_expression(step)?;
                    }
                    self.resolve_statement(body)?;
                    self.scopes.pop().unwrap();
                    Ok(())
                }
            }
        }

        pub fn resolve_declaration(&mut self, decl: &mut ast::Declaration) -> Result<(), Error> {
            let ast::Declaration { ident, exp } = decl;

            if self.current_scope().contains_key(&ident.data) {
                return Err(Error::VariableAlreadyDeclared(ident.clone()));
            }

            let unique_name = self.new_var(&ident.data);
            self.current_scope()
                .insert(ident.data.clone(), unique_name.clone());
            ident.data = unique_name;
            if let Some(exp) = exp {
                self.resolve_expression(exp)?;
            }
            Ok(())
        }

        pub fn resolve_expression(&mut self, exp: &mut ast::Expression) -> Result<(), Error> {
            match exp {
                ast::Expression::Constant(_) => Ok(()),
                ast::Expression::Unary { exp, .. } => self.resolve_expression(exp),
                ast::Expression::Binary { lhs, rhs, .. } => {
                    self.resolve_expression(lhs)?;
                    self.resolve_expression(rhs)?;
                    Ok(())
                }
                ast::Expression::Var(var) => {
                    if let Some(unique_name) = self.lookup(&var.data) {
                        var.data = unique_name.clone();
                        Ok(())
                    } else {
                        return Err(Error::VariableNotDeclared(var.clone()));
                    }
                }
                ast::Expression::Assignment { lhs, rhs } => {
                    if !matches!(lhs.as_ref(), ast::Expression::Var(_)) {
                        return Err(Error::InvalidLValue(lhs.as_ref().clone()));
                    }
                    self.resolve_expression(lhs)?;
                    self.resolve_expression(rhs)?;
                    Ok(())
                }
                ast::Expression::Conditional {
                    condition,
                    then_branch,
                    else_branch,
                } => {
                    self.resolve_expression(condition)?;
                    self.resolve_expression(then_branch)?;
                    self.resolve_expression(else_branch)?;
                    Ok(())
                }
            }
        }
    }
}

pub mod loop_label {
    use crate::{ast, span::HasSpan};
    use ecow::EcoString;

    #[derive(Debug, Default)]
    pub struct LoopLabel {
        loop_counter: usize,
    }

    #[derive(Debug, thiserror::Error)]
    pub enum Error {
        #[error("Break statement not in loop")]
        BreakNotInLoop(std::ops::Range<usize>),
        #[error("Continue statement not in loop")]
        ContinueNotInLoop(std::ops::Range<usize>),
    }

    impl HasSpan for Error {
        fn span(&self) -> std::ops::Range<usize> {
            match self {
                Error::BreakNotInLoop(span) => span.clone(),
                Error::ContinueNotInLoop(span) => span.clone(),
            }
        }
    }

    impl LoopLabel {
        fn new_label(&mut self) -> EcoString {
            let label = EcoString::from(format!("label.{}", self.loop_counter));
            self.loop_counter += 1;
            label
        }

        pub fn label_program(&mut self, program: &mut ast::Program) -> Result<(), Error> {
            self.label_block(None, &mut program.function_definition.body)
        }

        fn label_statement(
            &mut self,
            current_label: Option<EcoString>,
            stmt: &mut ast::Statement,
        ) -> Result<(), Error> {
            match stmt {
                ast::Statement::Return(_) => Ok(()),
                ast::Statement::Expression(_) => Ok(()),
                ast::Statement::If {
                    condition: _,
                    then_branch,
                    else_branch,
                } => {
                    self.label_statement(current_label.clone(), then_branch)?;
                    if let Some(else_branch) = else_branch {
                        self.label_statement(current_label, else_branch)?;
                    }
                    Ok(())
                }
                ast::Statement::Compound(block) => self.label_block(current_label, block),
                ast::Statement::Break { label, span } => {
                    if let Some(current_label) = current_label {
                        *label = current_label;
                    } else {
                        return Err(Error::BreakNotInLoop(span.clone()));
                    }
                    Ok(())
                }
                ast::Statement::Continue { label, span } => {
                    if let Some(current_label) = current_label {
                        *label = current_label;
                    } else {
                        return Err(Error::ContinueNotInLoop(span.clone()));
                    }
                    Ok(())
                }
                ast::Statement::While {
                    label,
                    condition: _,
                    body,
                } => {
                    let new_label = self.new_label();
                    *label = new_label.clone();
                    self.label_statement(Some(new_label.clone()), body)?;
                    Ok(())
                }
                ast::Statement::DoWhile {
                    label,
                    condition: _,
                    body,
                } => {
                    let new_label = self.new_label();
                    *label = new_label.clone();
                    self.label_statement(Some(new_label.clone()), body)?;
                    Ok(())
                }
                ast::Statement::For {
                    label,
                    init: _,
                    condition: _,
                    step: _,
                    body,
                } => {
                    let new_label = self.new_label();
                    *label = new_label.clone();
                    self.label_statement(Some(new_label.clone()), body)?;
                    Ok(())
                }
                ast::Statement::Null => Ok(()),
            }
        }

        fn label_block(
            &mut self,
            current_label: Option<EcoString>,
            block: &mut ast::Block,
        ) -> Result<(), Error> {
            for block_item in &mut block.0 {
                match block_item {
                    ast::BlockItem::Declaration(_) => {}
                    ast::BlockItem::Statement(stmt) => {
                        self.label_statement(current_label.clone(), stmt)?;
                    }
                }
            }
            Ok(())
        }
    }
}
