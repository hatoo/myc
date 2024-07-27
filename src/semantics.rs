pub use loop_label::LoopLabel;
pub use type_check::TypeChecker;
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
        scopes: Vec<HashMap<EcoString, VarInfo>>,
    }

    #[derive(Debug, Clone)]
    struct VarInfo {
        new_name: EcoString,
        has_linkage: bool,
    }

    #[derive(Debug, thiserror::Error)]
    pub enum Error {
        #[error("Variable not declared: {0}")]
        VariableNotDeclared(Spanned<EcoString>),
        #[error("Variable already declared: {0}")]
        VariableAlreadyDeclared(Spanned<EcoString>),
        #[error("Invalid lvalue: {0:?}")]
        InvalidLValue(Expression),
        #[error("Undeclared function: {0:?}")]
        UndeclaredFunction(Expression),
    }

    impl HasSpan for Error {
        fn span(&self) -> std::ops::Range<usize> {
            match self {
                Error::VariableNotDeclared(ident) => ident.span.clone(),
                Error::VariableAlreadyDeclared(ident) => ident.span.clone(),
                Error::InvalidLValue(exp) => exp.span(),
                Error::UndeclaredFunction(exp) => exp.span(),
            }
        }
    }

    impl VarResolver {
        pub fn new_var(&mut self, prefix: &EcoString) -> EcoString {
            let var = EcoString::from(format!("{}.{}", prefix, self.var_counter));
            self.var_counter += 1;
            var
        }

        fn lookup(&self, ident: &EcoString) -> Option<&VarInfo> {
            for scope in self.scopes.iter().rev() {
                if let Some(var_info) = scope.get(ident) {
                    return Some(var_info);
                }
            }
            None
        }

        fn current_scope(&mut self) -> &mut HashMap<EcoString, VarInfo> {
            self.scopes.last_mut().unwrap()
        }

        pub fn resolve_program(&mut self, program: &mut ast::Program) -> Result<(), Error> {
            self.scopes.push(HashMap::new());
            for decl in &mut program.decls {
                match decl {
                    ast::Declaration::VarDecl(decl) => self.resolve_var_decl_file_scope(decl)?,
                    ast::Declaration::FunDecl(decl) => self.resolve_fun_decl(decl, true)?,
                }
            }
            self.scopes.pop().unwrap();
            Ok(())
        }

        pub fn resolve_block_item(&mut self, block_item: &mut ast::BlockItem) -> Result<(), Error> {
            match block_item {
                ast::BlockItem::Declaration(decl) => self.resolve_decl(decl),
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
                            ast::ForInit::VarDecl(decl) => {
                                self.resolve_var_decl_local(decl)?;
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

        pub fn resolve_decl(&mut self, decl: &mut ast::Declaration) -> Result<(), Error> {
            match decl {
                ast::Declaration::VarDecl(decl) => self.resolve_var_decl_local(decl),
                ast::Declaration::FunDecl(decl) => self.resolve_fun_decl(decl, false),
            }
        }
        pub fn resolve_fun_decl(
            &mut self,
            decl: &mut ast::FunDecl,
            file_scope: bool,
        ) -> Result<(), Error> {
            let ast::FunDecl {
                name,
                params,
                body,
                storage_class,
            } = decl;

            if !file_scope && storage_class == &Some(ast::StorageClass::Static) {
                return Err(Error::VariableAlreadyDeclared(name.clone()));
            }

            if let Some(VarInfo {
                has_linkage: false, ..
            }) = self.current_scope().get(&name.data)
            {
                return Err(Error::VariableAlreadyDeclared(name.clone()));
            }

            self.current_scope().insert(
                name.data.clone(),
                VarInfo {
                    new_name: name.data.clone(),
                    has_linkage: true,
                },
            );

            self.scopes.push(HashMap::new());

            for param in params {
                let unique_name = self.new_var(&param.data);
                if self
                    .current_scope()
                    .insert(
                        param.data.clone(),
                        VarInfo {
                            new_name: unique_name.clone(),
                            has_linkage: false,
                        },
                    )
                    .is_some()
                {
                    return Err(Error::VariableAlreadyDeclared(param.clone()));
                }
                param.data = unique_name;
            }

            if let Some(body) = body {
                for block_item in &mut body.0 {
                    self.resolve_block_item(block_item)?;
                }
            }

            self.scopes.pop().unwrap();
            Ok(())
        }

        pub fn resolve_fun_decl_local(&mut self, decl: &mut ast::FunDecl) -> Result<(), Error> {
            let ast::FunDecl {
                name,
                params,
                body,
                storage_class: _,
            } = decl;

            if let Some(VarInfo {
                has_linkage: false, ..
            }) = self.current_scope().get(&name.data)
            {
                return Err(Error::VariableAlreadyDeclared(name.clone()));
            }

            self.current_scope().insert(
                name.data.clone(),
                VarInfo {
                    new_name: name.data.clone(),
                    has_linkage: true,
                },
            );

            self.scopes.push(HashMap::new());

            for param in params {
                let unique_name = self.new_var(&param.data);
                if self
                    .current_scope()
                    .insert(
                        param.data.clone(),
                        VarInfo {
                            new_name: unique_name.clone(),
                            has_linkage: false,
                        },
                    )
                    .is_some()
                {
                    return Err(Error::VariableAlreadyDeclared(param.clone()));
                }
                param.data = unique_name;
            }

            if let Some(body) = body {
                for block_item in &mut body.0 {
                    self.resolve_block_item(block_item)?;
                }
            }

            self.scopes.pop().unwrap();
            Ok(())
        }

        pub fn resolve_var_decl_file_scope(
            &mut self,
            decl: &mut ast::VarDecl,
        ) -> Result<(), Error> {
            let ast::VarDecl {
                ident,
                init: _,
                storage_class: _,
            } = decl;

            self.current_scope().insert(
                ident.data.clone(),
                VarInfo {
                    new_name: ident.data.clone(),
                    has_linkage: true,
                },
            );
            Ok(())
        }

        pub fn resolve_var_decl_local(&mut self, decl: &mut ast::VarDecl) -> Result<(), Error> {
            let ast::VarDecl {
                ident,
                init,
                storage_class,
            } = decl;

            if let Some(var) = self.current_scope().get(&ident.data) {
                if !(var.has_linkage && storage_class == &Some(ast::StorageClass::Extern)) {
                    return Err(Error::VariableAlreadyDeclared(ident.clone()));
                }
            }

            if storage_class == &Some(ast::StorageClass::Extern) {
                self.current_scope().insert(
                    ident.data.clone(),
                    VarInfo {
                        new_name: ident.data.clone(),
                        has_linkage: true,
                    },
                );
                Ok(())
            } else {
                let unique_name = self.new_var(&ident.data);
                self.current_scope().insert(
                    ident.data.clone(),
                    VarInfo {
                        new_name: unique_name.clone(),
                        has_linkage: false,
                    },
                );
                ident.data = unique_name;
                if let Some(exp) = init {
                    self.resolve_expression(exp)?;
                }
                Ok(())
            }
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
                        var.data = unique_name.new_name.clone();
                        Ok(())
                    } else {
                        Err(Error::VariableNotDeclared(var.clone()))
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
                ast::Expression::FunctionCall { name, args, .. } => {
                    if let Some(var_info) = self.lookup(&name.data) {
                        name.data = var_info.new_name.clone();
                        for arg in args {
                            self.resolve_expression(arg)?;
                        }
                        Ok(())
                    } else {
                        Err(Error::UndeclaredFunction(exp.clone()))
                    }
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
}

pub mod type_check {
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
                            crate::ast::ForInit::VarDecl(decl) => {
                                self.check_var_decl_local(decl)?
                            }
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
}
