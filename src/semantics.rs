use std::collections::HashMap;

use ecow::EcoString;

use crate::ast;

#[derive(Debug, Default)]
pub struct VarResolver {
    var_counter: usize,
    var_map: HashMap<EcoString, EcoString>,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Variable not declared: {0}")]
    VariableNotDeclared(EcoString),
    #[error("Variable already declared: {0}")]
    VariableAlreadyDeclared(EcoString),
    #[error("Invalid lvalue")]
    InvalidLValue,
}

impl VarResolver {
    pub fn new_var(&mut self, prefix: &EcoString) -> EcoString {
        let var = EcoString::from(format!("{}.{}", prefix, self.var_counter));
        self.var_counter += 1;
        var
    }

    pub fn resolve_program(&mut self, program: &mut ast::Program) -> Result<(), Error> {
        for block_item in &mut program.function_definition.body {
            self.resolve_block_item(block_item)?;
        }
        Ok(())
    }

    pub fn resolve_block_item(&mut self, block_item: &mut ast::BlockItem) -> Result<(), Error> {
        match block_item {
            ast::BlockItem::Declaration(decl) => self.resolve_declaration(decl),
            ast::BlockItem::Statement(stmt) => self.resolve_statements(stmt),
        }
    }

    pub fn resolve_statements(&mut self, stmts: &mut ast::Statement) -> Result<(), Error> {
        match stmts {
            ast::Statement::Return(decl) => self.resolve_expression(decl),
            ast::Statement::Expression(exp) => self.resolve_expression(exp),
            ast::Statement::Null => Ok(()),
        }
    }

    pub fn resolve_declaration(&mut self, decl: &mut ast::Declaration) -> Result<(), Error> {
        let ast::Declaration { ident, exp } = decl;

        if self.var_map.contains_key(ident) {
            return Err(Error::VariableAlreadyDeclared(ident.clone()));
        }

        let unique_name = self.new_var(ident);
        self.var_map.insert(ident.clone(), unique_name.clone());
        *ident = unique_name.clone();
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
            ast::Expression::Var(ident) => {
                if let Some(unique_name) = self.var_map.get(ident) {
                    *ident = unique_name.clone();
                    Ok(())
                } else {
                    return Err(Error::VariableNotDeclared(ident.clone()));
                }
            }
            ast::Expression::Assignment { lhs, rhs } => {
                if !matches!(lhs.as_ref(), ast::Expression::Var(_)) {
                    return Err(Error::InvalidLValue);
                }
                self.resolve_expression(lhs)?;
                self.resolve_expression(rhs)?;
                Ok(())
            }
        }
    }
}