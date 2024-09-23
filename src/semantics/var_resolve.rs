use std::collections::HashMap;

use ecow::EcoString;

use crate::{
    ast::{self, Expression, StructDecl, UnionDecl},
    lexer::{HasTokenSpan, TokenSpanned},
};

#[derive(Debug, Clone, Copy)]
enum TypeKind {
    Struct,
    Union,
}

#[derive(Debug, Default)]
struct Scope {
    vars: HashMap<EcoString, VarInfo>,
    types: HashMap<EcoString, (EcoString, TypeKind)>,
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
    #[error("Typedef not declared: {0}")]
    TypedefNotDeclared(TokenSpanned<EcoString>),
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
            Error::TypedefNotDeclared(ident) => ident.span.clone(),
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

    fn lookup_type(&self, ident: &EcoString) -> Option<(bool, &EcoString, TypeKind)> {
        for (i, Scope { types, .. }) in self.scopes.iter().rev().enumerate() {
            if let Some((new_name, kind)) = types.get(ident) {
                return Some((i == 0, new_name, *kind));
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

    fn current_scope_type(&mut self) -> &mut HashMap<EcoString, (EcoString, TypeKind)> {
        &mut self.scopes.last_mut().unwrap().types
    }

    pub fn resolve_program(&mut self, program: &mut ast::Program) -> Result<(), Error> {
        self.push();
        for decl in &mut program.decls {
            match decl {
                ast::Declaration::Var(decl) => self.resolve_var_decl_file_scope(decl)?,
                ast::Declaration::Fun(decl) => self.resolve_fun_decl(decl, true)?,
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
                            self.resolve_var_decl_local(decl, false)?;
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
            ast::Declaration::Var(decl) => self.resolve_var_decl_local(decl, false),
            ast::Declaration::Fun(decl) => self.resolve_fun_decl(decl, false),
        }
    }
    fn resolve_fun_decl(&mut self, decl: &mut ast::FunDecl, file_scope: bool) -> Result<(), Error> {
        let ast::FunDecl {
            type_decl_ret,
            type_decl_params,
            name,
            params,
            body,
            storage_class,
            ty,
        } = decl;

        if let Some(type_decl) = type_decl_ret {
            self.resolve_type_declaration(type_decl, false)?;
        }

        self.resolve_var_type(&mut ty.ret, name.span.clone())?;

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

        for ((param, decl), param_ty) in params
            .iter_mut()
            .zip(type_decl_params.iter_mut())
            .zip(ty.params.iter_mut())
        {
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
                && (!param.data.is_empty() || body.is_some())
            {
                return Err(Error::VariableAlreadyDeclared(param.clone()));
            }
            param.data = unique_name;
            self.push();
            if let Some(type_decl) = decl {
                self.resolve_type_declaration(type_decl, false)?;
            }
            self.resolve_var_type(param_ty, name.span.clone())?;
            self.pop();
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
        let is_type_only = decl.is_type_only();
        let ast::VarDecl {
            type_decl,
            ident,
            init: _,
            storage_class: _,
            ty,
        } = decl;

        if let Some(type_decl) = type_decl {
            self.resolve_type_declaration(type_decl, is_type_only)?;
        }
        self.resolve_var_type(ty, ident.span.clone())?;

        if let TokenSpanned {
            data: Some(ident), ..
        } = &ident
        {
            self.current_scope_var().insert(
                ident.clone(),
                VarInfo {
                    new_name: ident.clone(),
                    has_linkage: true,
                },
            );
        }
        Ok(())
    }

    fn resolve_var_decl_local(
        &mut self,
        decl: &mut ast::VarDecl,
        look_up_only: bool,
    ) -> Result<(), Error> {
        let is_type_only = decl.is_type_only();
        let ast::VarDecl {
            type_decl,
            ident,
            init,
            storage_class,
            ty,
        } = decl;

        if let Some(type_decl) = type_decl {
            self.resolve_type_declaration(type_decl, !look_up_only && is_type_only)?;
        }
        self.resolve_var_type(ty, ident.span.clone())?;

        if let TokenSpanned {
            data: Some(ident),
            span,
        } = ident
        {
            let old_ident = ident;
            let ident = TokenSpanned {
                data: old_ident.clone(),
                span: span.clone(),
            };
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
            } else if storage_class == &Some(ast::StorageClass::Typedef) {
                let new_name = self.new_var(&ident.data);
                self.current_scope_var().insert(
                    ident.data.clone(),
                    VarInfo {
                        new_name: new_name.clone(),
                        has_linkage: false,
                    },
                );
                *old_ident = new_name;
            } else {
                let unique_name = self.new_var(&ident.data);
                self.current_scope_var().insert(
                    ident.data.clone(),
                    VarInfo {
                        new_name: unique_name.clone(),
                        has_linkage: false,
                    },
                );
                *old_ident = unique_name;
                if let Some(init) = init {
                    self.resolve_initializer(init)?;
                }
            }
        }

        Ok(())
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
                self.resolve_var_decl_local(target, true)?;
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
                self.resolve_var_decl_local(&mut ty.data, true)?;
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
                if let Some((_, new_name, TypeKind::Struct)) = self.lookup_type(name) {
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
                if let Some((_, new_name, TypeKind::Union)) = self.lookup_type(name) {
                    *name = new_name.clone();
                    Ok(())
                } else {
                    Err(Error::StructNotDeclared(TokenSpanned {
                        data: name.clone(),
                        span,
                    }))
                }
            }
            ast::VarType::Typedef(name) => {
                if let Some(var_info) = self.lookup_var(&name.data) {
                    name.data = var_info.new_name.clone();
                    Ok(())
                } else {
                    Err(Error::TypedefNotDeclared(TokenSpanned {
                        data: name.data.clone(),
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

    fn resolve_type_declaration(
        &mut self,
        decl: &mut ast::TypeDeclaration,
        force_decl: bool,
    ) -> Result<(), Error> {
        match decl {
            ast::TypeDeclaration::Struct(decl) => {
                self.resolve_structure_declaration(decl, force_decl)
            }
            ast::TypeDeclaration::Union(decl) => self.resolve_union_declaration(decl, force_decl),
            ast::TypeDeclaration::Fun { ret, params } => {
                if let Some(ret) = ret {
                    self.resolve_type_declaration(ret, force_decl)?;
                }

                for param in params {
                    if let Some(param) = param {
                        self.resolve_type_declaration(param, force_decl)?;
                    }
                }

                Ok(())
            }
        }
    }

    fn resolve_structure_declaration(
        &mut self,
        decl: &mut StructDecl,
        force_decl: bool,
    ) -> Result<(), Error> {
        let StructDecl { tag, member_decls } = decl;

        if member_decls.is_empty() && !force_decl {
            match self.lookup_type(&tag.data) {
                None => {
                    let new_name = self.new_var(&tag.data);
                    self.current_scope_type()
                        .insert(tag.data.clone(), (new_name.clone(), TypeKind::Struct));
                    tag.data = new_name.clone();
                }
                Some((_, prev, TypeKind::Struct)) => {
                    tag.data = prev.clone();
                }
                _ => {
                    todo!()
                }
            }
        } else {
            match self.lookup_type(&tag.data) {
                None | Some((false, _, _)) => {
                    let new_name = self.new_var(&tag.data);
                    self.current_scope_type()
                        .insert(tag.data.clone(), (new_name.clone(), TypeKind::Struct));
                    tag.data = new_name.clone();
                }
                Some((true, prev, TypeKind::Struct)) => {
                    tag.data = prev.clone();
                }
                Some((true, _, _)) => {
                    return Err(Error::StructNotDeclared(tag.clone()));
                }
            }

            for member_decl in member_decls {
                if let Some(type_decl) = &mut member_decl.type_decl {
                    self.resolve_type_declaration(type_decl, false)?;
                }
                self.resolve_var_type(&mut member_decl.ty, tag.span.clone())?;
            }
        }

        Ok(())
    }

    fn resolve_union_declaration(
        &mut self,
        decl: &mut UnionDecl,
        force_decl: bool,
    ) -> Result<(), Error> {
        let UnionDecl { tag, member_decls } = decl;

        if member_decls.is_empty() && !force_decl {
            match self.lookup_type(&tag.data) {
                None => {
                    let new_name = self.new_var(&tag.data);
                    self.current_scope_type()
                        .insert(tag.data.clone(), (new_name.clone(), TypeKind::Union));
                    tag.data = new_name.clone();
                }
                Some((_, prev, TypeKind::Union)) => {
                    tag.data = prev.clone();
                }
                _ => {
                    todo!()
                }
            }
        } else {
            match self.lookup_type(&tag.data) {
                None | Some((false, _, _)) => {
                    let new_name = self.new_var(&tag.data);
                    self.current_scope_type()
                        .insert(tag.data.clone(), (new_name.clone(), TypeKind::Union));
                    tag.data = new_name.clone();
                }
                Some((true, prev, TypeKind::Union)) => {
                    tag.data = prev.clone();
                }
                Some((true, _, _)) => {
                    return Err(Error::StructNotDeclared(tag.clone()));
                }
            }

            for member_decl in member_decls {
                if let Some(type_decl) = &mut member_decl.type_decl {
                    self.resolve_type_declaration(type_decl, false)?;
                }
                self.resolve_var_type(&mut member_decl.ty, tag.span.clone())?;
            }
        }

        Ok(())
    }
}
