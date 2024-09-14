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
                match item {
                    BlockItem::Statement(stmt) => check_statement(stmt, known_labels)?,
                    _ => {}
                }
            }
        }
        Statement::Goto(label) => {
            if !known_labels.contains(&label.data) {
                return Err(Error::GotoUndefined(label.clone()));
            }
        }
        _ => {}
    }

    Ok(())
}

fn check_block(block: &Block) -> Result<(), Error> {
    let known_labels = collect_labels(block)?;

    for item in &block.0 {
        match item {
            BlockItem::Statement(stmt) => {
                check_statement(stmt, &known_labels)?;
            }
            _ => {}
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
            let new_labels = collect_labels(block)?;
            known_labels.extend(new_labels);
        }
        _ => {}
    }

    Ok(())
}

fn collect_labels(block: &Block) -> Result<HashSet<EcoString>, Error> {
    let mut set = HashSet::new();

    for item in &block.0 {
        match item {
            BlockItem::Statement(stmt) => collect_statement(stmt, &mut set)?,
            _ => {}
        }
    }

    Ok(set)
}
