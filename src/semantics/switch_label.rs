use crate::ast::{self, BlockItem, Statement, SwitchLabels};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Case label not in switch")]
    CaseNotInSwitch(std::ops::Range<usize>),
    #[error("Default label not in switch")]
    DefaultNotInSwitch(std::ops::Range<usize>),
    #[error("Duplicated default label")]
    DuplicatedDefaultLabel(std::ops::Range<usize>),
}

pub fn collect_switch_labels(program: &mut ast::Program) -> Result<(), Error> {
    for decl in &mut program.decls {
        match decl {
            ast::Declaration::VarDecl(_) => {}
            ast::Declaration::StructDecl(_) => {}
            ast::Declaration::FunDecl(fun_decl) => {
                if let Some(body) = &mut fun_decl.body {
                    for item in &mut body.0 {
                        match item {
                            BlockItem::Declaration(_) => {}
                            BlockItem::Statement(statement) => {
                                collect_switch_labels_statement(statement, &mut None)?;
                            }
                        }
                    }
                }
            }
        }
    }

    Ok(())
}
pub fn collect_switch_labels_statement(
    statement: &mut ast::Statement,
    labels: &mut Option<SwitchLabels>,
) -> Result<(), Error> {
    match statement {
        Statement::Case {
            exp,
            statement,
            label,
            span,
        } => {
            if let Some(labels) = labels {
                labels.cases.push((exp.clone(), label.clone()));
            } else {
                return Err(Error::CaseNotInSwitch(span.clone()));
            }
            collect_switch_labels_statement(statement, labels)?;
        }
        Statement::Default {
            statement,
            label,
            span,
        } => {
            if let Some(labels) = labels {
                if labels.default.is_some() {
                    return Err(Error::DuplicatedDefaultLabel(span.clone()));
                }
                labels.default = Some(label.clone());
            } else {
                return Err(Error::DefaultNotInSwitch(span.clone()));
            }
            collect_switch_labels_statement(statement, labels)?;
        }
        Statement::Switch {
            statement, labels, ..
        } => {
            let mut new_labels = Some(SwitchLabels::default());
            collect_switch_labels_statement(statement, &mut new_labels)?;
            *labels = new_labels.unwrap();
        }

        Statement::Break { .. } => {}
        Statement::Compound(block) => {
            for item in &mut block.0 {
                match item {
                    BlockItem::Declaration(_) => {}
                    BlockItem::Statement(statement) => {
                        collect_switch_labels_statement(statement, labels)?;
                    }
                }
            }
        }
        Statement::Continue { .. } => {}
        Statement::DoWhile { body, .. } => {
            collect_switch_labels_statement(body, labels)?;
        }
        Statement::Expression(_) => {}
        Statement::For { body, .. } => {
            collect_switch_labels_statement(body, labels)?;
        }
        Statement::Goto(_) => {}
        Statement::If {
            then_branch,
            else_branch,
            ..
        } => {
            collect_switch_labels_statement(then_branch, labels)?;
            if let Some(else_branch) = else_branch {
                collect_switch_labels_statement(else_branch, labels)?;
            }
        }
        Statement::Label { statement, .. } => {
            collect_switch_labels_statement(statement, labels)?;
        }
        Statement::Null => {}
        Statement::Return(_) => {}
        Statement::While { body, .. } => {
            collect_switch_labels_statement(body, labels)?;
        }
    }

    Ok(())
}
