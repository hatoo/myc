use std::collections::HashSet;

use ecow::EcoString;

use crate::{
    ast::{Block, BlockItem, Program, Statement},
    lexer::{HasTokenSpan, TokenSpanned},
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("GoTo undefined: {0}")]
    GotoUndefined(TokenSpanned<EcoString>),
    #[error("Duplicate label: {0}")]
    DuplicateLabel(TokenSpanned<EcoString>),
}

impl HasTokenSpan for Error {
    fn token_span(&self) -> std::ops::Range<usize> {
        match self {
            Error::GotoUndefined(label) => label.span.clone(),
            Error::DuplicateLabel(label) => label.span.clone(),
        }
    }
}

pub fn check_goto(program: &Program) -> Result<(), Error> {
    for decl in &program.decls {
        match decl {
            crate::ast::Declaration::VarDecl(_) => {}
            crate::ast::Declaration::StructDecl(_) => {}
            crate::ast::Declaration::FunDecl(fun_decl) => {
                if let Some(body) = &fun_decl.body {
                    check_block(body)?;
                }
            }
        }
    }

    Ok(())
}

fn check_statement(statement: &Statement, known_labels: &HashSet<EcoString>) -> Result<(), Error> {
    match statement {
        Statement::Label { statement, .. } => {
            check_statement(statement, known_labels)?;
        }
        Statement::If {
            condition: _,
            then_branch,
            else_branch,
        } => {
            check_statement(then_branch, known_labels)?;
            if let Some(else_branch) = else_branch {
                check_statement(else_branch, known_labels)?;
            }
        }
        Statement::While {
            label: _,
            condition: _,
            body,
        } => {
            check_statement(body, known_labels)?;
        }
        Statement::For { body, .. } => {
            check_statement(body, known_labels)?;
        }
        Statement::DoWhile { body, .. } => {
            check_statement(body, known_labels)?;
        }
        Statement::Compound(block) => {
            for item in &block.0 {
                if let BlockItem::Statement(stmt) = item {
                    check_statement(stmt, known_labels)?
                }
            }
        }
        Statement::Goto(label) => {
            if !known_labels.contains(&label.data) {
                return Err(Error::GotoUndefined(label.clone()));
            }
        }
        Statement::Default { statement, .. } => {
            check_statement(statement, known_labels)?;
        }
        Statement::Case { statement, .. } => {
            check_statement(statement, known_labels)?;
        }
        Statement::Switch { statement, .. } => {
            check_statement(statement, known_labels)?;
        }
        _ => {}
    }

    Ok(())
}

fn check_block(block: &Block) -> Result<(), Error> {
    let mut known_labels = HashSet::new();
    collect_labels(block, &mut known_labels)?;

    for item in &block.0 {
        if let BlockItem::Statement(stmt) = item {
            check_statement(stmt, &known_labels)?;
        }
    }

    Ok(())
}

fn collect_statement(
    statement: &Statement,
    known_labels: &mut HashSet<EcoString>,
) -> Result<(), Error> {
    match statement {
        Statement::Label { label, statement } => {
            if !known_labels.insert(label.data.clone()) {
                return Err(Error::DuplicateLabel(label.clone()));
            }
            collect_statement(statement, known_labels)?;
        }
        Statement::If {
            condition: _,
            then_branch,
            else_branch,
        } => {
            collect_statement(then_branch, known_labels)?;
            if let Some(else_branch) = else_branch {
                collect_statement(else_branch, known_labels)?;
            }
        }
        Statement::While {
            label: _,
            condition: _,
            body,
        } => {
            collect_statement(body, known_labels)?;
        }
        Statement::For { body, .. } => {
            collect_statement(body, known_labels)?;
        }
        Statement::DoWhile { body, .. } => {
            collect_statement(body, known_labels)?;
        }
        Statement::Compound(block) => {
            collect_labels(block, known_labels)?;
        }
        Statement::Default { statement, .. } => {
            collect_statement(statement, known_labels)?;
        }
        Statement::Case { statement, .. } => {
            collect_statement(statement, known_labels)?;
        }
        Statement::Switch { statement, .. } => {
            collect_statement(statement, known_labels)?;
        }
        _ => {}
    }

    Ok(())
}

fn collect_labels(block: &Block, known_labels: &mut HashSet<EcoString>) -> Result<(), Error> {
    for item in &block.0 {
        if let BlockItem::Statement(stmt) = item {
            collect_statement(stmt, known_labels)?
        }
    }

    Ok(())
}
