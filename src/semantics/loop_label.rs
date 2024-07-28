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
        for decl in &mut program.decls {
            match decl {
                ast::Declaration::VarDecl(_) => {}
                ast::Declaration::FunDecl(fun_decl) => {
                    if let Some(body) = &mut fun_decl.body {
                        self.label_block(None, body)?;
                    }
                }
            }
        }
        Ok(())
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
