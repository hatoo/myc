use std::collections::{hash_map::Entry, HashMap};

use ecow::EcoString;

use crate::{
    ast::{self, Expression, VarType},
    span::{HasSpan, Spanned},
};

pub type SymbolTable = HashMap<EcoString, Attr>;

#[derive(Debug, Default)]
pub struct TypeChecker {
    pub sym_table: SymbolTable,
}

#[derive(Debug)]
pub enum Attr {
    Fun {
        ty: ast::FunType,
        defined: bool,
        global: bool,
    },
    Static {
        ty: ast::VarType,
        init: InitialValue,
        global: bool,
    },
    Local(ast::VarType),
}

impl Attr {
    pub fn ty(&self) -> &ast::VarType {
        match self {
            Attr::Fun { ty, .. } => &ty.ret,
            Attr::Static { ty, .. } => ty,
            Attr::Local(ty) => ty,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum InitialValue {
    Tentative,
    Initial(StaticInit),
    NoInitializer,
}

#[derive(Debug, Clone, Copy)]
pub enum StaticInit {
    Int(i32),
    Long(i64),
    Uint(u32),
    Ulong(u64),
    Double(f64),
}

impl StaticInit {
    pub fn alignment(&self) -> usize {
        match self {
            StaticInit::Int(_) => 4,
            StaticInit::Uint(_) => 4,
            StaticInit::Long(_) => 8,
            StaticInit::Ulong(_) => 8,
            StaticInit::Double(_) => 8,
        }
    }

    pub fn size(&self) -> usize {
        match self {
            StaticInit::Int(_) => 4,
            StaticInit::Uint(_) => 4,
            StaticInit::Long(_) => 8,
            StaticInit::Ulong(_) => 8,
            StaticInit::Double(_) => 8,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Incompatible types: {0:?}")]
    IncompatibleTypes(std::ops::Range<usize>),
    #[error("Function redefined: {0}")]
    Redefined(Spanned<EcoString>),
    #[error("Static function declaration follows non-static : {0}")]
    StaticFunAfterNonStatic(Spanned<EcoString>),
    #[error("Bad Initializer")]
    BadInitializer(Spanned<EcoString>),
    #[error("Incompatible linkage: {0}")]
    IncompatibleLinkage(Spanned<EcoString>),
    #[error("Function declaration in block scope has a body")]
    BlockScopeFunWithBody(Spanned<EcoString>),
    #[error("For loop init must not has storage class")]
    BadForInit(Spanned<EcoString>),
}

impl HasSpan for Error {
    fn span(&self) -> std::ops::Range<usize> {
        match self {
            Error::IncompatibleTypes(span) => span.clone(),
            Error::Redefined(ident) => ident.span.clone(),
            Error::StaticFunAfterNonStatic(ident) => ident.span.clone(),
            Error::BadInitializer(ident) => ident.span.clone(),
            Error::IncompatibleLinkage(ident) => ident.span.clone(),
            Error::BlockScopeFunWithBody(ident) => ident.span.clone(),
            Error::BadForInit(ident) => ident.span.clone(),
        }
    }
}

fn common_pointer_type<'a>(
    e0: &'a ast::Expression,
    e1: &'a ast::Expression,
) -> Option<&'a ast::VarType> {
    let ty0 = e0.ty();
    let ty1 = e1.ty();

    if ty0 == ty1 {
        Some(ty0)
    } else if e0.is_null_pointer_constant() {
        Some(ty1)
    } else if e1.is_null_pointer_constant() {
        Some(ty0)
    } else {
        None
    }
}

fn common_type(ty0: ast::VarType, ty1: ast::VarType) -> ast::VarType {
    if ty0 == ast::VarType::Double || ty1 == ast::VarType::Double {
        return ast::VarType::Double;
    }

    if ty0 == ty1 {
        ty0
    } else if ty0.size() == ty1.size() {
        if ty0.is_signed() {
            ty1
        } else {
            ty0
        }
    } else if ty0.size() > ty1.size() {
        ty0
    } else {
        ty1
    }
}

fn convert_to(exp: &mut ast::Expression, ty: &ast::VarType) {
    if exp.ty() != ty {
        *exp = ast::Expression::Cast {
            exp: Box::new(exp.clone()),
            target: ty.clone(),
        };
    }
}

fn convert_by_assignment(exp: &mut ast::Expression, ty: &ast::VarType) -> Result<(), Error> {
    let ety = exp.ty();

    if ety == ty {
        return Ok(());
    }

    if !ety.is_pointer() && !ty.is_pointer() {
        convert_to(exp, ty);
        return Ok(());
    }

    if exp.is_null_pointer_constant() && ty.is_pointer() {
        convert_to(exp, ty);
        return Ok(());
    }

    Err(Error::IncompatibleTypes(exp.span()))
}

impl TypeChecker {
    pub fn check_program(&mut self, program: &mut crate::ast::Program) -> Result<(), Error> {
        for decl in &mut program.decls {
            match decl {
                crate::ast::Declaration::VarDecl(decl) => self.check_var_decl_file(decl)?,
                crate::ast::Declaration::FunDecl(decl) => self.check_fun_decl(decl)?,
            }
        }

        Ok(())
    }

    fn check_fun_decl(&mut self, fun_decl: &mut crate::ast::FunDecl) -> Result<(), Error> {
        let crate::ast::FunDecl {
            name,
            params,
            body,
            storage_class,
            ty,
        } = fun_decl;

        let mut new_global = storage_class != &Some(crate::ast::StorageClass::Static);
        let mut already_defined = false;

        if let Some(attr) = self.sym_table.get(&name.data) {
            if let Attr::Fun {
                defined,
                global,
                ty: ty0,
            } = attr
            {
                if ty0 != ty {
                    return Err(Error::IncompatibleTypes(name.span.clone()));
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
                return Err(Error::IncompatibleTypes(name.span.clone()));
            }
        }

        self.sym_table.insert(
            name.data.clone(),
            Attr::Fun {
                defined: already_defined || body.is_some(),
                global: new_global,
                ty: ty.clone(),
            },
        );

        if let Some(body) = body {
            for (param, ty) in params.iter().zip(ty.params.iter()) {
                self.sym_table
                    .insert(param.data.clone(), Attr::Local(ty.clone()));
            }
            self.check_block_local(body, &ty.ret)?;
        }

        Ok(())
    }

    fn check_block_local(
        &mut self,
        block: &mut crate::ast::Block,
        ret_type: &VarType,
    ) -> Result<(), Error> {
        for block_item in &mut block.0 {
            match block_item {
                crate::ast::BlockItem::Declaration(decl) => self.check_decl_local(decl)?,
                crate::ast::BlockItem::Statement(stmt) => self.check_statement(stmt, ret_type)?,
            }
        }
        Ok(())
    }

    fn check_decl_local(&mut self, decl: &mut crate::ast::Declaration) -> Result<(), Error> {
        match decl {
            crate::ast::Declaration::VarDecl(decl) => self.check_var_decl_local(decl),
            crate::ast::Declaration::FunDecl(decl) => {
                if decl.body.is_some() {
                    return Err(Error::BlockScopeFunWithBody(decl.name.clone()));
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
            ty,
        } = decl;

        let mut init = match init {
            Some(Expression::Constant(Spanned { data: c, .. })) => InitialValue::Initial(
                c.get_static_init(ty)
                    .ok_or_else(|| Error::BadInitializer(ident.clone()))?,
            ),
            Some(_) => return Err(Error::BadInitializer(ident.clone())),
            None => {
                if storage_class == &Some(crate::ast::StorageClass::Extern) {
                    InitialValue::NoInitializer
                } else {
                    InitialValue::Tentative
                }
            }
        };

        let mut global = storage_class != &Some(crate::ast::StorageClass::Static);

        match self.sym_table.get(&ident.data) {
            Some(Attr::Fun { .. }) => {
                return Err(Error::IncompatibleTypes(ident.span.clone()));
            }
            Some(Attr::Static {
                init: old_init,
                global: old_global,
                ty: old_ty,
            }) => {
                if storage_class == &Some(crate::ast::StorageClass::Extern) {
                    global = *old_global;
                } else if *old_global != global {
                    return Err(Error::IncompatibleLinkage(ident.clone()));
                }

                if ty != old_ty {
                    return Err(Error::IncompatibleTypes(ident.span.clone()));
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
            Some(Attr::Local(_)) => {
                unreachable!()
            }
            None => {}
        }

        self.sym_table.insert(
            ident.data.clone(),
            Attr::Static {
                init,
                global,
                ty: ty.clone(),
            },
        );
        Ok(())
    }

    fn check_var_decl_local(&mut self, decl: &mut crate::ast::VarDecl) -> Result<(), Error> {
        let crate::ast::VarDecl {
            ident,
            init,
            storage_class,
            ty,
        } = decl;

        match storage_class {
            Some(crate::ast::StorageClass::Extern) => {
                if init.is_some() {
                    return Err(Error::BadInitializer(ident.clone()));
                }
                match self.sym_table.entry(ident.data.clone()) {
                    Entry::Occupied(o) => match o.get() {
                        Attr::Fun { .. } => {
                            return Err(Error::IncompatibleTypes(ident.span.clone()));
                        }
                        Attr::Local(ty0) | Attr::Static { ty: ty0, .. } => {
                            if ty0 != ty {
                                return Err(Error::IncompatibleTypes(ident.span.clone()));
                            }
                        }
                    },
                    Entry::Vacant(v) => {
                        v.insert(Attr::Static {
                            init: InitialValue::NoInitializer,
                            global: true,
                            ty: ty.clone(),
                        });
                    }
                }
            }
            Some(crate::ast::StorageClass::Static) => {
                let init = match init {
                    Some(Expression::Constant(val)) => InitialValue::Initial(
                        val.data
                            .get_static_init(ty)
                            .ok_or_else(|| Error::BadInitializer(ident.clone()))?,
                    ),
                    None => InitialValue::Initial(ty.zero()),
                    _ => return Err(Error::BadInitializer(ident.clone())),
                };
                self.sym_table.insert(
                    ident.data.clone(),
                    Attr::Static {
                        init,
                        global: false,
                        ty: ty.clone(),
                    },
                );
            }
            _ => {
                self.sym_table
                    .insert(ident.data.clone(), Attr::Local(ty.clone()));
                if let Some(exp) = init {
                    self.check_expression(exp)?;
                    convert_by_assignment(exp, ty)?;
                }
            }
        }

        Ok(())
    }

    fn check_expression(
        &mut self,
        exp: &mut crate::ast::Expression,
    ) -> Result<ast::VarType, Error> {
        match exp {
            crate::ast::Expression::Var(name, ty) => match self.sym_table.get(&name.data) {
                Some(Attr::Fun { ty: fty, .. }) => {
                    // extra credit
                    let t = ast::VarType::Pointer(Box::new(ast::Ty::Fun(fty.clone())));
                    *ty = t.clone();
                    Ok(t)
                }
                Some(Attr::Static { ty: target, .. }) | Some(Attr::Local(target)) => {
                    *ty = target.clone();
                    Ok(target.clone())
                }
                None => Err(Error::IncompatibleTypes(name.span.clone())),
            },
            crate::ast::Expression::Constant(_) => Ok(exp.ty().clone()),
            crate::ast::Expression::Unary { op, exp, ty } => {
                match op.data {
                    ast::UnaryOp::Not => {
                        self.check_expression(exp)?;
                        *ty = ast::VarType::Int;
                    }
                    ast::UnaryOp::Complement => {
                        *ty = self.check_expression(exp)?;
                        if *ty == ast::VarType::Double || ty.is_pointer() {
                            return Err(Error::IncompatibleTypes(exp.span()));
                        }
                    }
                    ast::UnaryOp::Negate => {
                        *ty = self.check_expression(exp)?;
                        if ty.is_pointer() {
                            return Err(Error::IncompatibleTypes(exp.span()));
                        }
                    }
                }
                Ok(ty.clone())
            }
            crate::ast::Expression::Binary { op, lhs, rhs, ty } => {
                let tyl = self.check_expression(lhs)?;
                let tyr = self.check_expression(rhs)?;

                match op {
                    ast::BinaryOp::And | ast::BinaryOp::Or => {
                        *ty = ast::VarType::Int;
                    }
                    ast::BinaryOp::Equal | ast::BinaryOp::NotEqual => {
                        let cty = if tyl.is_pointer() || tyr.is_pointer() {
                            if let Some(cty) = common_pointer_type(lhs, rhs) {
                                cty.clone()
                            } else {
                                return Err(Error::IncompatibleTypes(exp.span()));
                            }
                        } else {
                            common_type(tyl, tyr)
                        };

                        convert_to(lhs, &cty);
                        convert_to(rhs, &cty);

                        *ty = ast::VarType::Int;
                    }
                    _ => {
                        let cty = common_type(tyl, tyr);
                        convert_to(lhs, &cty);
                        convert_to(rhs, &cty);

                        match op {
                            ast::BinaryOp::Add | ast::BinaryOp::Subtract => {
                                *ty = cty;
                            }
                            ast::BinaryOp::Multiply | ast::BinaryOp::Divide => {
                                if cty.is_pointer() {
                                    return Err(Error::IncompatibleTypes(exp.span()));
                                }
                                *ty = cty;
                            }
                            ast::BinaryOp::Remainder => {
                                if cty == ast::VarType::Double || cty.is_pointer() {
                                    return Err(Error::IncompatibleTypes(exp.span()));
                                }
                                *ty = cty;
                            }
                            _ => {
                                *ty = ast::VarType::Int;
                            }
                        }
                    }
                }

                Ok(ty.clone())
            }
            crate::ast::Expression::Assignment { lhs, rhs } => {
                if !lhs.is_lvalue() {
                    return Err(Error::IncompatibleTypes(lhs.span()));
                }
                let tyl = self.check_expression(lhs)?;
                self.check_expression(rhs)?;
                convert_by_assignment(rhs, &tyl)?;
                Ok(tyl)
            }
            crate::ast::Expression::Conditional {
                condition,
                then_branch,
                else_branch,
            } => {
                self.check_expression(condition)?;
                let tyl = self.check_expression(then_branch)?;
                let tyr = self.check_expression(else_branch)?;

                let cty = if tyl.is_pointer() || tyr.is_pointer() {
                    if let Some(cty) = common_pointer_type(then_branch, else_branch) {
                        cty.clone()
                    } else {
                        return Err(Error::IncompatibleTypes(exp.span()));
                    }
                } else {
                    common_type(tyl, tyr)
                };

                convert_to(then_branch, &cty);
                convert_to(else_branch, &cty);

                Ok(cty)
            }
            crate::ast::Expression::FunctionCall {
                name,
                args,
                ty: fty,
            } => match self.sym_table.get(&name.data) {
                Some(Attr::Fun { ty, .. }) => {
                    if ty.params.len() != args.len() {
                        return Err(Error::IncompatibleTypes(name.span.clone()));
                    }
                    let ret = ty.ret.clone();

                    for (arg, ty) in args.iter_mut().zip(ty.params.clone().into_iter()) {
                        self.check_expression(arg)?;
                        convert_by_assignment(arg, &ty)?;
                    }
                    *fty = ret.clone();
                    Ok(ret.clone())
                }
                Some(
                    Attr::Local(ast::VarType::Pointer(pty))
                    | Attr::Static {
                        ty: ast::VarType::Pointer(pty),
                        ..
                    },
                ) => {
                    if let ast::Ty::Fun(ty) = pty.as_ref() {
                        if ty.params.len() != args.len() {
                            return Err(Error::IncompatibleTypes(name.span.clone()));
                        }
                        let ret = ty.ret.clone();

                        for (arg, ty) in args.iter_mut().zip(ty.params.clone().into_iter()) {
                            self.check_expression(arg)?;
                            convert_by_assignment(arg, &ty)?;
                        }
                        *fty = ret.clone();
                        Ok(ret.clone())
                    } else {
                        Err(Error::IncompatibleTypes(name.span.clone()))
                    }
                }
                _ => Err(Error::IncompatibleTypes(name.span.clone())),
            },
            crate::ast::Expression::Cast { target, exp } => {
                let ty = self.check_expression(exp)?;

                if (target.is_pointer() && ty == VarType::Double)
                    || (ty.is_pointer() && target == &VarType::Double)
                {
                    return Err(Error::IncompatibleTypes(exp.span()));
                }

                Ok(target.clone())
            }
            crate::ast::Expression::Dereference(exp) => {
                let ty = self.check_expression(exp)?;
                if let ast::VarType::Pointer(ty) = ty {
                    match ty.as_ref() {
                        ast::Ty::Fun(_) => Err(Error::IncompatibleTypes(exp.span())),
                        ast::Ty::Var(ty) => Ok(ty.clone()),
                    }
                } else {
                    Err(Error::IncompatibleTypes(exp.span()))
                }
            }
            ast::Expression::AddrOf { exp, ty } => {
                if !exp.is_lvalue() {
                    return Err(Error::IncompatibleTypes(exp.span()));
                }

                let exp_ty = self.check_expression(exp)?;
                *ty = ast::VarType::Pointer(Box::new(ast::Ty::Var(exp_ty.clone())));
                Ok(ty.clone())
            }
        }
    }

    fn check_statement(
        &mut self,
        stmt: &mut crate::ast::Statement,
        ret_type: &ast::VarType,
    ) -> Result<(), Error> {
        match stmt {
            crate::ast::Statement::Return(exp) => {
                self.check_expression(exp)?;
                convert_by_assignment(exp, ret_type)?;
                Ok(())
            }
            crate::ast::Statement::Expression(exp) => {
                self.check_expression(exp)?;
                Ok(())
            }
            crate::ast::Statement::If {
                condition,
                then_branch,
                else_branch,
            } => {
                self.check_expression(condition)?;
                self.check_statement(then_branch, ret_type)?;
                if let Some(else_branch) = else_branch {
                    self.check_statement(else_branch, ret_type)?;
                }
                Ok(())
            }
            crate::ast::Statement::Compound(block) => {
                self.check_block_local(block, ret_type)?;
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
                self.check_statement(body, ret_type)?;
                Ok(())
            }
            crate::ast::Statement::DoWhile {
                label: _,
                condition,
                body,
            } => {
                self.check_statement(body, ret_type)?;
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
                            if decl.storage_class.is_some() {
                                return Err(Error::BadForInit(decl.ident.clone()));
                            }
                            self.check_var_decl_local(decl)?;
                        }
                        crate::ast::ForInit::Expression(exp) => {
                            self.check_expression(exp)?;
                        }
                    }
                }
                if let Some(condition) = condition {
                    self.check_expression(condition)?;
                }
                if let Some(step) = step {
                    self.check_expression(step)?;
                }
                self.check_statement(body, ret_type)?;
                Ok(())
            }
            crate::ast::Statement::Null => Ok(()),
        }
    }
}
