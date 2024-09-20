use std::collections::HashMap;

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

#[derive(Debug, Default)]
pub struct GotoCheck {
    counter: usize,
}

impl GotoCheck {
    pub fn check_goto(&mut self, program: &mut Program) -> Result<(), Error> {
        for decl in &mut program.decls {
            match decl {
                crate::ast::Declaration::VarDecl(_) => {}
                crate::ast::Declaration::StructDecl(_) => {}
                crate::ast::Declaration::UnionDecl(_) => {}
                crate::ast::Declaration::FunDecl(fun_decl) => {
                    if let Some(body) = &mut fun_decl.body {
                        self.check_block(body)?;
                    }
                }
            }
        }

        Ok(())
    }
    fn new_label(&mut self, prefix: EcoString) -> EcoString {
        let label = EcoString::from(format!(".L{}.{}", prefix, self.counter));
        self.counter += 1;
        label
    }

    fn collect_label_statement(
        &mut self,
        statement: &mut Statement,
        known_labels: &mut HashMap<EcoString, EcoString>,
    ) -> Result<(), Error> {
        match statement {
            Statement::Label { label, statement } => {
                let new_label = self.new_label(label.data.clone());
                if known_labels
                    .insert(label.data.clone(), new_label.clone())
                    .is_some()
                {
                    return Err(Error::DuplicateLabel(label.clone()));
                }
                label.data = new_label;
                self.collect_label_statement(statement, known_labels)?;
            }
            Statement::If {
                condition: _,
                then_branch,
                else_branch,
            } => {
                self.collect_label_statement(then_branch, known_labels)?;
                if let Some(else_branch) = else_branch {
                    self.collect_label_statement(else_branch, known_labels)?;
                }
            }
            Statement::While {
                label: _,
                condition: _,
                body,
            } => {
                self.collect_label_statement(body, known_labels)?;
            }
            Statement::For { body, .. } => {
                self.collect_label_statement(body, known_labels)?;
            }
            Statement::DoWhile { body, .. } => {
                self.collect_label_statement(body, known_labels)?;
            }
            Statement::Compound(block) => {
                self.collect_labels(block, known_labels)?;
            }
            Statement::Default { statement, .. } => {
                self.collect_label_statement(statement, known_labels)?;
            }
            Statement::Case { statement, .. } => {
                self.collect_label_statement(statement, known_labels)?;
            }
            Statement::Switch { statement, .. } => {
                self.collect_label_statement(statement, known_labels)?;
            }
            _ => {}
        }

        Ok(())
    }

    fn collect_labels(
        &mut self,
        block: &mut Block,
        known_labels: &mut HashMap<EcoString, EcoString>,
    ) -> Result<(), Error> {
        for item in &mut block.0 {
            if let BlockItem::Statement(stmt) = item {
                self.collect_label_statement(stmt, known_labels)?
            }
        }

        Ok(())
    }

    fn check_block(&mut self, block: &mut Block) -> Result<(), Error> {
        let mut known_labels = HashMap::new();
        self.collect_labels(block, &mut known_labels)?;

        for item in &mut block.0 {
            if let BlockItem::Statement(stmt) = item {
                check_statement(stmt, &known_labels)?;
            }
        }

        Ok(())
    }
}

fn check_statement(
    statement: &mut Statement,
    known_labels: &HashMap<EcoString, EcoString>,
) -> Result<(), Error> {
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
            for item in &mut block.0 {
                if let BlockItem::Statement(stmt) = item {
                    check_statement(stmt, known_labels)?
                }
            }
        }
        Statement::Goto(label) => {
            if let Some(new_label) = known_labels.get(&label.data) {
                label.data = new_label.clone();
            } else {
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
