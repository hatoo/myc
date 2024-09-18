use std::collections::HashMap;

use ecow::EcoString;

use crate::{
    ast::{self, Expression, StructDecl, UnionDecl},
    lexer::{HasTokenSpan, TokenSpanned},
};

#[derive(Debug, Default)]
struct Scope {
    vars: HashMap<EcoString, VarInfo>,
    structs: HashMap<EcoString, EcoString>,
    unions: HashMap<EcoString, EcoString>,
}

#[derive(Debug, Default)]
pub struct VarResolver {
    var_counter: usize,
    scopes: Vec<Scope>,
}

#[derive(Debug, Clone)]
struct VarInfo {
    new_name: EcoString,
    has_linkage: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Variable not declared: {0}")]
    VariableNotDeclared(TokenSpanned<EcoString>),
    #[error("Variable already declared: {0}")]
    VariableAlreadyDeclared(TokenSpanned<EcoString>),
    #[error("Invalid lvalue: {0:?}")]
    InvalidLValue(Expression),
    #[error("Undeclared function: {0:?}")]
    UndeclaredFunction(Expression),
    #[error("Static function declaration in block scope: {0}")]
    StaticFunInBlock(TokenSpanned<EcoString>),
    #[error("Struct not declared: {0}")]
    StructNotDeclared(TokenSpanned<EcoString>),
}

impl HasTokenSpan for Error {
    fn token_span(&self) -> std::ops::Range<usize> {
        match self {
            Error::VariableNotDeclared(ident) => ident.span.clone(),
            Error::VariableAlreadyDeclared(ident) => ident.span.clone(),
            Error::InvalidLValue(exp) => exp.token_span(),
            Error::UndeclaredFunction(exp) => exp.token_span(),
            Error::StaticFunInBlock(ident) => ident.span.clone(),
            Error::StructNotDeclared(ident) => ident.span.clone(),
        }
    }
}

impl VarResolver {
    fn new_var(&mut self, prefix: &EcoString) -> EcoString {
        let var = EcoString::from(format!("{}.{}", prefix, self.var_counter));
        self.var_counter += 1;
        var
    }

    fn lookup_var(&self, ident: &EcoString) -> Option<&VarInfo> {
        for Scope { vars, .. } in self.scopes.iter().rev() {
            if let Some(var_info) = vars.get(ident) {
                return Some(var_info);
            }
        }
        None
    }

    fn lookup_struct(&self, ident: &EcoString) -> Option<(bool, &EcoString)> {
        for (i, Scope { structs, .. }) in self.scopes.iter().rev().enumerate() {
            if let Some(new_name) = structs.get(ident) {
                return Some((i == 0, new_name));
            }
        }
        None
    }

    fn lookup_union(&self, ident: &EcoString) -> Option<(bool, &EcoString)> {
        for (i, Scope { unions, .. }) in self.scopes.iter().rev().enumerate() {
            if let Some(new_name) = unions.get(ident) {
                return Some((i == 0, new_name));
            }
        }
        None
    }

    fn push(&mut self) {
        self.scopes.push(Default::default());
    }

    fn pop(&mut self) {
        self.scopes.pop().unwrap();
    }

    fn current_scope_var(&mut self) -> &mut HashMap<EcoString, VarInfo> {
        &mut self.scopes.last_mut().unwrap().vars
    }

    fn current_scope_struct(&mut self) -> &mut HashMap<EcoString, EcoString> {
        &mut self.scopes.last_mut().unwrap().structs
    }

    fn current_scope_union(&mut self) -> &mut HashMap<EcoString, EcoString> {
        &mut self.scopes.last_mut().unwrap().unions
    }

    pub fn resolve_program(&mut self, program: &mut ast::Program) -> Result<(), Error> {
        self.push();
        for decl in &mut program.decls {
            match decl {
                ast::Declaration::VarDecl(decl) => self.resolve_var_decl_file_scope(decl)?,
                ast::Declaration::FunDecl(decl) => self.resolve_fun_decl(decl, true)?,
                ast::Declaration::StructDecl(decl) => self.resolve_structure_declaration(decl)?,
                ast::Declaration::UnionDecl(decl) => self.resolve_union_declaration(decl)?,
            }
        }
        self.pop();
        Ok(())
    }

    fn resolve_block_item(&mut self, block_item: &mut ast::BlockItem) -> Result<(), Error> {
        match block_item {
            ast::BlockItem::Declaration(decl) => self.resolve_decl(decl),
            ast::BlockItem::Statement(stmt) => self.resolve_statement(stmt),
        }
    }

    fn resolve_statement(&mut self, stmt: &mut ast::Statement) -> Result<(), Error> {
        match stmt {
            ast::Statement::Return(decl) => {
                if let Some(decl) = decl {
                    self.resolve_expression(decl)?;
                }
                Ok(())
            }
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
                self.push();
                for block_item in &mut stmts.0 {
                    self.resolve_block_item(block_item)?;
                }
                self.pop();
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
                self.push();
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
                self.pop();
                Ok(())
            }
            ast::Statement::Goto(_) => Ok(()),
            ast::Statement::Label {
                statement: stmt, ..
            } => self.resolve_statement(stmt),
            ast::Statement::Case { exp, statement, .. } => {
                self.resolve_expression(exp)?;
                self.resolve_statement(statement)?;
                Ok(())
            }
            ast::Statement::Default { statement, .. } => {
                self.resolve_statement(statement)?;
                Ok(())
            }
            ast::Statement::Switch { exp, statement, .. } => {
                self.resolve_expression(exp)?;
                self.resolve_statement(statement)?;
                Ok(())
            }
        }
    }

    fn resolve_decl(&mut self, decl: &mut ast::Declaration) -> Result<(), Error> {
        match decl {
            ast::Declaration::VarDecl(decl) => self.resolve_var_decl_local(decl),
            ast::Declaration::FunDecl(decl) => self.resolve_fun_decl(decl, false),
            ast::Declaration::StructDecl(decl) => self.resolve_structure_declaration(decl),
            ast::Declaration::UnionDecl(decl) => self.resolve_union_declaration(decl),
        }
    }
    fn resolve_fun_decl(&mut self, decl: &mut ast::FunDecl, file_scope: bool) -> Result<(), Error> {
        let ast::FunDecl {
            name,
            params,
            body,
            storage_class,
            ty,
        } = decl;

        self.resolve_fun_type(ty, name.span.clone())?;

        if !file_scope && storage_class == &Some(ast::StorageClass::Static) {
            return Err(Error::StaticFunInBlock(name.clone()));
        }

        if let Some(VarInfo {
            has_linkage: false, ..
        }) = self.current_scope_var().get(&name.data)
        {
            return Err(Error::VariableAlreadyDeclared(name.clone()));
        }

        self.current_scope_var().insert(
            name.data.clone(),
            VarInfo {
                new_name: name.data.clone(),
                has_linkage: true,
            },
        );

        self.push();

        for param in params {
            let unique_name = self.new_var(&param.data);
            if self
                .current_scope_var()
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

        self.pop();
        Ok(())
    }

    fn resolve_var_decl_file_scope(&mut self, decl: &mut ast::VarDecl) -> Result<(), Error> {
        let ast::VarDecl {
            ident,
            init: _,
            storage_class: _,
            ty,
        } = decl;

        self.resolve_var_type(ty, ident.span.clone())?;

        self.current_scope_var().insert(
            ident.data.clone(),
            VarInfo {
                new_name: ident.data.clone(),
                has_linkage: true,
            },
        );
        Ok(())
    }

    fn resolve_var_decl_local(&mut self, decl: &mut ast::VarDecl) -> Result<(), Error> {
        let ast::VarDecl {
            ident,
            init,
            storage_class,
            ty,
        } = decl;

        self.resolve_var_type(ty, ident.span.clone())?;

        if let Some(var) = self.current_scope_var().get(&ident.data) {
            if !(var.has_linkage && storage_class == &Some(ast::StorageClass::Extern)) {
                return Err(Error::VariableAlreadyDeclared(ident.clone()));
            }
        }

        if storage_class == &Some(ast::StorageClass::Extern) {
            self.current_scope_var().insert(
                ident.data.clone(),
                VarInfo {
                    new_name: ident.data.clone(),
                    has_linkage: true,
                },
            );
            Ok(())
        } else {
            let unique_name = self.new_var(&ident.data);
            self.current_scope_var().insert(
                ident.data.clone(),
                VarInfo {
                    new_name: unique_name.clone(),
                    has_linkage: false,
                },
            );
            ident.data = unique_name;
            if let Some(init) = init {
                self.resolve_initializer(init)?;
            }
            Ok(())
        }
    }

    fn resolve_initializer(&mut self, init: &mut ast::Initializer) -> Result<(), Error> {
        match init {
            ast::Initializer::SingleInit(exp) => self.resolve_expression(exp),
            ast::Initializer::CompoundInit(inits) => {
                for init in inits {
                    self.resolve_initializer(init)?;
                }
                Ok(())
            }
        }
    }

    fn resolve_expression(&mut self, exp: &mut ast::Expression) -> Result<(), Error> {
        match exp {
            ast::Expression::Constant(_) => Ok(()),
            ast::Expression::Unary { exp, .. } => self.resolve_expression(exp),
            ast::Expression::Binary { lhs, rhs, .. } => {
                self.resolve_expression(lhs)?;
                self.resolve_expression(rhs)?;
                Ok(())
            }
            ast::Expression::Var(var, ..) => {
                if let Some(unique_name) = self.lookup_var(&var.data) {
                    var.data = unique_name.new_name.clone();
                    Ok(())
                } else {
                    Err(Error::VariableNotDeclared(var.clone()))
                }
            }
            ast::Expression::Assignment { lhs, rhs } => {
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
            ast::Expression::FunctionCall { callee, args, .. } => {
                self.resolve_expression(callee)?;
                for arg in args {
                    self.resolve_expression(arg)?;
                }
                Ok(())
            }
            ast::Expression::Cast { target, exp } => {
                self.resolve_var_type(target, exp.token_span())?;
                self.resolve_expression(exp)?;
                Ok(())
            }
            ast::Expression::AddrOf { exp, .. } => {
                self.resolve_expression(exp)?;
                Ok(())
            }
            ast::Expression::Dereference(exp) => {
                self.resolve_expression(exp)?;
                Ok(())
            }
            ast::Expression::Subscript { array, index, .. } => {
                self.resolve_expression(array)?;
                self.resolve_expression(index)?;
                Ok(())
            }
            ast::Expression::String(..) => Ok(()),
            ast::Expression::Sizeof(exp) => {
                self.resolve_expression(exp)?;
                Ok(())
            }
            ast::Expression::SizeofType(ty) => {
                self.resolve_var_type(&mut ty.data, ty.span.clone())?;
                Ok(())
            }
            ast::Expression::Dot { structure, .. } => {
                self.resolve_expression(structure)?;
                Ok(())
            }
            ast::Expression::Arrow { pointer, .. } => {
                self.resolve_expression(pointer)?;
                Ok(())
            }
            ast::Expression::Increment { exp, .. } => {
                self.resolve_expression(exp)?;
                Ok(())
            }
            ast::Expression::Decrement { exp, .. } => {
                self.resolve_expression(exp)?;
                Ok(())
            }
        }
    }

    fn resolve_fun_type(
        &mut self,
        ty: &mut ast::FunType,
        span: std::ops::Range<usize>,
    ) -> Result<(), Error> {
        for arg in &mut ty.params {
            self.resolve_var_type(arg, span.clone())?;
        }
        self.resolve_var_type(&mut ty.ret, span)?;

        Ok(())
    }

    fn resolve_var_type(
        &mut self,
        ty: &mut ast::VarType,
        span: std::ops::Range<usize>,
    ) -> Result<(), Error> {
        match ty {
            ast::VarType::Struct(name) => {
                if let Some((_, new_name)) = self.lookup_struct(name) {
                    *name = new_name.clone();
                    Ok(())
                } else {
                    Err(Error::StructNotDeclared(TokenSpanned {
                        data: name.clone(),
                        span,
                    }))
                }
            }
            ast::VarType::Union(name) => {
                if let Some((_, new_name)) = self.lookup_union(name) {
                    *name = new_name.clone();
                    Ok(())
                } else {
                    Err(Error::StructNotDeclared(TokenSpanned {
                        data: name.clone(),
                        span,
                    }))
                }
            }
            ast::VarType::Pointer(inner) => match inner.as_mut() {
                ast::Ty::Fun(ty) => self.resolve_fun_type(ty, span),
                ast::Ty::Var(ty) => self.resolve_var_type(ty, span),
            },
            ast::VarType::Array { element, .. } => {
                self.resolve_var_type(element.as_mut(), span)?;
                Ok(())
            }
            _ => Ok(()),
        }
    }

    fn resolve_structure_declaration(&mut self, decl: &mut StructDecl) -> Result<(), Error> {
        let StructDecl { tag, member_decls } = decl;

        match self.lookup_struct(&tag.data) {
            None | Some((false, _)) => {
                let new_name = self.new_var(&tag.data);
                self.current_scope_struct()
                    .insert(tag.data.clone(), new_name.clone());
                tag.data = new_name.clone();
            }
            Some((true, prev)) => {
                tag.data = prev.clone();
            }
        }

        for member_decl in member_decls {
            self.resolve_var_type(&mut member_decl.ty, tag.span.clone())?;
        }

        Ok(())
    }

    fn resolve_union_declaration(&mut self, decl: &mut UnionDecl) -> Result<(), Error> {
        let UnionDecl { tag, member_decls } = decl;

        match self.lookup_union(&tag.data) {
            None | Some((false, _)) => {
                let new_name = self.new_var(&tag.data);
                self.current_scope_union()
                    .insert(tag.data.clone(), new_name.clone());
                tag.data = new_name.clone();
            }
            Some((true, prev)) => {
                tag.data = prev.clone();
            }
        }

        for member_decl in member_decls {
            self.resolve_var_type(&mut member_decl.ty, tag.span.clone())?;
        }

        Ok(())
    }
}
