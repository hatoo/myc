use core::panic;
use std::{
    collections::{hash_map::Entry, BTreeMap, HashMap, HashSet},
    fmt::Display,
    ops::DerefMut,
};

use ecow::EcoString;

use crate::{
    ast::{self, BaseType, Expression, Initializer, Ty, VarType},
    lexer::{HasTokenSpan, MayHasTokenSpan, TokenSpanned},
    math::round_up,
};

#[derive(Debug, Default)]
pub struct SymbolTable(pub HashMap<EcoString, Attr>);

impl std::ops::Deref for SymbolTable {
    type Target = HashMap<EcoString, Attr>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl DerefMut for SymbolTable {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl SymbolTable {
    pub fn struct_def(&self, tag: &EcoString) -> &StructDef {
        if let Attr::Struct(def) = self.get(tag).unwrap() {
            def
        } else {
            unreachable!("{}", tag)
        }
    }

    pub fn union_def(&self, tag: &EcoString) -> &UnionDef {
        if let Attr::Union(def) = self.get(tag).unwrap() {
            def
        } else {
            unreachable!("{}", tag)
        }
    }

    pub fn flatten(&self, ty: &VarType) -> Vec<BaseType> {
        let mut ret = Vec::new();

        match ty {
            VarType::Base(base) => {
                ret.push(*base);
            }
            VarType::Pointer(_) => {
                ret.push(BaseType::Ulong);
            }
            VarType::Array { element, size } => {
                for _ in 0..*size {
                    ret.extend(self.flatten(element));
                }
            }
            VarType::Struct(name) => {
                let structure = self.struct_def(name);

                for member in &structure.members {
                    ret.extend(self.flatten(&member.ty));
                }
            }
            VarType::Union(name) => {
                let union = self.union_def(name);

                let (mem, max_align) = union
                    .members
                    .iter()
                    .map(|m| (m, self.alignment(&m.ty)))
                    .max_by_key(|(_, a)| *a)
                    .unwrap();

                let elems = union.size / max_align;

                let flat = self.flatten(&mem.ty);

                for _ in 0..elems {
                    ret.extend(flat.clone());
                }
            }
            VarType::Void => unreachable!(),
        }

        ret
    }

    pub fn ty_size(&self, ty: &ast::Ty) -> usize {
        match ty {
            ast::Ty::Var(ref ty) => self.size(ty),
            ast::Ty::Fun(_) => 1,
        }
    }

    pub fn size(&self, ty: &ast::VarType) -> usize {
        match ty {
            ast::VarType::Array { element, size } => {
                let element_size = self.size(element);
                element_size * size
            }
            ast::VarType::Struct(name) => {
                let StructDef { size, .. } = self.struct_def(name);
                *size
            }
            ast::VarType::Union(name) => {
                let UnionDef { size, .. } = self.union_def(name);
                *size
            }
            ast::VarType::Pointer(_) => 8,
            ast::VarType::Base(base) => base.size(),
            ast::VarType::Void => panic!("Get size of void"),
        }
    }

    pub fn alignment(&self, ty: &ast::VarType) -> usize {
        match ty {
            ast::VarType::Array { element, .. } => {
                if self.size(ty) < 16 {
                    self.alignment(element)
                } else {
                    16
                }
            }
            ast::VarType::Struct(name) => {
                let StructDef { alignment, .. } = self.struct_def(name);

                *alignment
            }
            ast::VarType::Union(name) => {
                let UnionDef { alignment, .. } = self.union_def(name);

                *alignment
            }
            ast::VarType::Pointer(_) => 8,
            ast::VarType::Base(base) => base.alignment(),
            ast::VarType::Void => panic!("Get alignment of void"),
        }
    }

    pub fn is_complete(&self, ty: &ast::VarType) -> bool {
        match ty {
            ast::VarType::Void => false,
            ast::VarType::Struct(tag) => {
                matches!(self.get(tag), Some(Attr::Struct(_)))
            }
            ast::VarType::Union(tag) => {
                matches!(self.get(tag), Some(Attr::Union(_)))
            }
            _ => true,
        }
    }

    pub fn is_pointer_to_complete(&self, ty: &ast::VarType) -> bool {
        match ty {
            ast::VarType::Pointer(ty) => match ty.as_ref() {
                ast::Ty::Var(ty) => self.is_complete(ty),
                _ => false,
            },
            _ => false,
        }
    }

    pub fn zero_init(&self, ty: &ast::VarType) -> ast::Initializer {
        match ty {
            VarType::Void => panic!("Zero init void"),
            VarType::Base(base) => ast::Initializer::zero_base(*base),
            VarType::Pointer(_) => ast::Initializer::SingleInit(ast::Expression::Constant(
                TokenSpanned::new_null(ast::Const::Ulong(0)),
            )),
            VarType::Array { element, size } => {
                let inits = vec![self.zero_init(element); *size];
                ast::Initializer::CompoundInit(inits)
            }
            VarType::Struct(tag) => {
                let StructDef { members, .. } = self.struct_def(tag);
                let inits = members
                    .iter()
                    .map(|member| self.zero_init(&member.ty))
                    .collect();
                ast::Initializer::CompoundInit(inits)
            }
            VarType::Union(tag) => {
                let UnionDef { members, .. } = self.union_def(tag);
                let (mem, max_align) = members
                    .iter()
                    .map(|m| (m, self.alignment(&m.ty)))
                    .max_by_key(|(_, a)| *a)
                    .unwrap();

                let elems = self.size(ty) / max_align;

                let inits = vec![self.zero_init(&mem.ty); elems];
                ast::Initializer::CompoundInit(inits)
            }
        }
    }
}

#[derive(Debug, Default)]
pub struct TypeChecker {
    pub tmp_string_count: usize,
    pub sym_table: SymbolTable,
}

#[derive(Debug, Clone)]
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
    Constant {
        ty: ast::VarType,
        init: StaticInit,
    },
    Local(ast::VarType),
    Struct(StructDef),
    Union(UnionDef),
}

#[derive(Debug, Clone)]
pub struct StructDef {
    pub alignment: usize,
    pub size: usize,
    pub members: Vec<StructMember>,
}

#[derive(Debug, Clone)]
pub struct UnionDef {
    pub alignment: usize,
    pub size: usize,
    pub members: Vec<UnionMember>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StructMember {
    pub name: EcoString,
    pub ty: ast::VarType,
    pub offset: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UnionMember {
    pub name: EcoString,
    pub ty: ast::VarType,
}

#[derive(Debug, Clone)]
pub enum InitialValue {
    Tentative,
    Initial(Vec<StaticInit>),
    NoInitializer,
}

#[derive(Debug, Clone)]
pub enum StaticInit {
    Int(i32),
    Long(i64),
    Uint(u32),
    Ulong(u64),
    Double(f64),
    Zero(usize),
    Char(i8),
    UChar(u8),
    String { data: Vec<u8>, pad: usize },
    Pointer(EcoString),
}

impl StaticInit {
    pub fn is_zero(&self) -> bool {
        matches!(
            self,
            StaticInit::Int(0)
                | StaticInit::Long(0)
                | StaticInit::Uint(0)
                | StaticInit::Ulong(0)
                | StaticInit::Double(0.0)
                | StaticInit::Zero(_)
        )
    }

    pub fn size(&self) -> usize {
        match self {
            StaticInit::Char(_) => 1,
            StaticInit::UChar(_) => 1,
            StaticInit::Int(_) => 4,
            StaticInit::Uint(_) => 4,
            StaticInit::Long(_) => 8,
            StaticInit::Ulong(_) => 8,
            StaticInit::Double(_) => 8,
            StaticInit::Zero(size) => *size,
            StaticInit::String { data, pad } => data.len() + *pad,
            StaticInit::Pointer(_) => 8,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Incompatible types: {0:?}")]
    IncompatibleTypes(std::ops::Range<usize>),
    #[error("Function redefined: {0}")]
    Redefined(TokenSpanned<EcoString>),
    #[error("Static function declaration follows non-static : {0}")]
    StaticFunAfterNonStatic(TokenSpanned<EcoString>),
    #[error("Bad Initializer")]
    BadInitializer(TokenSpanned<EcoString>),
    #[error("Incompatible linkage: {0}")]
    IncompatibleLinkage(TokenSpanned<EcoString>),
    #[error("Function declaration in block scope has a body")]
    BlockScopeFunWithBody(TokenSpanned<EcoString>),
    #[error("For loop init must not has storage class")]
    BadForInit(TokenSpanned<EcoString>),
    #[error("Case expression is not constant")]
    CaseExpIsNotConstant(std::ops::Range<usize>),
}

impl HasTokenSpan for Error {
    fn token_span(&self) -> std::ops::Range<usize> {
        match self {
            Error::IncompatibleTypes(span) => span.clone(),
            Error::Redefined(ident) => ident.span.clone(),
            Error::StaticFunAfterNonStatic(ident) => ident.span.clone(),
            Error::BadInitializer(ident) => ident.span.clone(),
            Error::IncompatibleLinkage(ident) => ident.span.clone(),
            Error::BlockScopeFunWithBody(ident) => ident.span.clone(),
            Error::BadForInit(ident) => ident.span.clone(),
            Error::CaseExpIsNotConstant(span) => span.clone(),
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
    } else if e1.is_null_pointer_constant()
        || ty0 == &ast::VarType::Pointer(Box::new(ast::Ty::Var(ast::VarType::Void)))
            && ty1.is_pointer()
    {
        Some(ty0)
    } else if ty1 == &ast::VarType::Pointer(Box::new(ast::Ty::Var(ast::VarType::Void)))
        && ty0.is_pointer()
    {
        Some(ty1)
    } else {
        None
    }
}

fn common_base_type(mut ty0: ast::BaseType, mut ty1: ast::BaseType) -> ast::BaseType {
    if ty0 == ast::BaseType::Double || ty1 == ast::BaseType::Double {
        return ast::BaseType::Double;
    }

    if ty0.is_character() {
        ty0 = ast::BaseType::Int
    }

    if ty1.is_character() {
        ty1 = ast::BaseType::Int
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

    if ety == &ast::VarType::Void {
        return Err(Error::IncompatibleTypes(exp.token_span()));
    }

    if ety.is_struct() || ty.is_struct() {
        return Err(Error::IncompatibleTypes(exp.token_span()));
    }

    if ety.is_union() || ty.is_union() {
        return Err(Error::IncompatibleTypes(exp.token_span()));
    }

    if !ety.is_pointer() && !ty.is_pointer() {
        convert_to(exp, ty);
        return Ok(());
    }

    if exp.is_null_pointer_constant() && ty.is_pointer() {
        convert_to(exp, ty);
        return Ok(());
    }

    if ty == &ast::VarType::Pointer(Box::new(ast::Ty::Var(ast::VarType::Void))) && ety.is_pointer()
    {
        convert_to(exp, ty);
        return Ok(());
    }

    if ty.is_pointer() && ety == &ast::VarType::Pointer(Box::new(ast::Ty::Var(ast::VarType::Void)))
    {
        convert_to(exp, ty);
        return Ok(());
    }

    Err(Error::IncompatibleTypes(exp.token_span()))
}

impl TypeChecker {
    pub fn check_program(&mut self, program: &mut crate::ast::Program) -> Result<(), Error> {
        for decl in &mut program.decls {
            match decl {
                crate::ast::Declaration::Var(decl) => self.check_var_decl_file(decl)?,
                crate::ast::Declaration::Fun(decl) => self.check_fun_decl(decl)?,
                crate::ast::Declaration::Struct(decl) => self.check_struct_decl(decl)?,
                crate::ast::Declaration::Union(decl) => self.check_union_decl(decl)?,
            }
        }

        Ok(())
    }

    fn tmp_string(&mut self) -> EcoString {
        let s = format!("string.{}", self.tmp_string_count);
        self.tmp_string_count += 1;
        s.into()
    }

    fn static_init_from_initializer(
        &mut self,
        target: &ast::VarType,
        inits: &Initializer,
    ) -> Result<Vec<StaticInit>, Error> {
        match inits {
            Initializer::SingleInit(Expression::String(s, _)) if target.is_array() => {
                let ast::VarType::Array { element, size } = target else {
                    unreachable!()
                };

                if !element.is_character() {
                    return Err(Error::IncompatibleTypes(0..0));
                }

                if size < &s.data.len() {
                    return Err(Error::IncompatibleTypes(0..0));
                }

                Ok(vec![StaticInit::String {
                    data: s.data.clone(),
                    pad: size - s.data.len(),
                }])
            }
            Initializer::SingleInit(exp) => {
                if let ast::VarType::Array { .. } = target {
                    return Err(Error::IncompatibleTypes(exp.token_span()));
                }

                Ok(vec![self.static_init(target, exp)?])
            }
            Initializer::CompoundInit(inits) => match target {
                ast::VarType::Array { element, size } => {
                    if inits.len() > *size {
                        return Err(Error::IncompatibleTypes(0..0));
                    }
                    let mut res = Vec::new();

                    for init in inits {
                        res.extend(self.static_init_from_initializer(element, init)?);
                    }

                    let filled = res.iter().map(|init| init.size()).sum::<usize>();
                    let target_size = self.sym_table.size(target);
                    if filled < target_size {
                        res.push(StaticInit::Zero(target_size - filled));
                    }

                    Ok(res)
                }
                ast::VarType::Struct(tag) => {
                    let StructDef { members, .. } = self.sym_table.struct_def(tag);
                    if inits.len() > members.len() {
                        return Err(Error::IncompatibleTypes(0..0));
                    }
                    let members = members.clone();
                    let mut res = Vec::new();

                    let mut offset = 0;
                    for (init, member) in inits.iter().zip(members.iter()) {
                        if member.offset > offset {
                            res.push(StaticInit::Zero(member.offset - offset));
                        }
                        res.extend(self.static_init_from_initializer(&member.ty, init)?);
                        offset = member.offset + self.sym_table.size(&member.ty);
                    }

                    let target_size = self.sym_table.size(target);
                    if offset < target_size {
                        res.push(StaticInit::Zero(target_size - offset));
                    }

                    Ok(res)
                }
                ast::VarType::Union(tag) => {
                    let UnionDef { members, size, .. } = self.sym_table.union_def(tag);
                    let size = *size;

                    if inits.len() > 1 {
                        return Err(Error::IncompatibleTypes(0..0));
                    }

                    let ty = members[0].ty.clone();
                    let mut res = self.static_init_from_initializer(&ty, &inits[0])?;
                    let ty_size = self.sym_table.size(&ty);
                    if size > ty_size {
                        res.push(StaticInit::Zero(size - ty_size));
                        Ok(res)
                    } else {
                        Ok(res)
                    }
                }

                _ => Err(Error::IncompatibleTypes(0..0)),
            },
        }
    }

    fn static_init(
        &mut self,
        target: &ast::VarType,
        exp: &Expression,
    ) -> Result<StaticInit, Error> {
        match exp {
            ast::Expression::Constant(TokenSpanned { data, .. }) => data
                .get_static_init(target)
                .ok_or_else(|| Error::IncompatibleTypes(exp.token_span())),

            ast::Expression::String(TokenSpanned { data, .. }, _) => match target {
                ast::VarType::Array { element, size } => {
                    if !element.is_character() {
                        return Err(Error::IncompatibleTypes(exp.token_span()));
                    }

                    if data.len() > *size {
                        return Err(Error::IncompatibleTypes(exp.token_span()));
                    }

                    Ok(StaticInit::String {
                        data: data.clone(),
                        pad: size - data.len(),
                    })
                }
                ast::VarType::Pointer(ty) => {
                    if let ast::Ty::Var(VarType::Base(BaseType::Char)) = ty.as_ref() {
                        let s = self.tmp_string();
                        self.sym_table.insert(
                            s.clone(),
                            Attr::Constant {
                                ty: ast::VarType::Array {
                                    element: Box::new(BaseType::Char.into()),
                                    size: data.len() + 1,
                                },
                                init: StaticInit::String {
                                    data: data.clone(),
                                    pad: 1,
                                },
                            },
                        );

                        Ok(StaticInit::Pointer(s))
                    } else {
                        Err(Error::IncompatibleTypes(exp.token_span()))
                    }
                }
                _ => Err(Error::IncompatibleTypes(exp.token_span())),
            },
            _ => Err(Error::IncompatibleTypes(exp.token_span())),
        }
    }

    fn check_fun_decl(&mut self, fun_decl: &mut crate::ast::FunDecl) -> Result<(), Error> {
        let crate::ast::FunDecl {
            name,
            params,
            body,
            storage_class,
            ty,
        } = fun_decl;

        self.validate_fun_type(ty, body.is_none())
            .map_err(|_| Error::IncompatibleTypes(name.span.clone()))?;

        for ty in &mut ty.params {
            if let ast::VarType::Array { element, .. } = ty {
                *ty = ast::VarType::Pointer(Box::new(ast::Ty::Var(element.as_ref().clone())));
            }
        }

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
            crate::ast::Declaration::Var(decl) => self.check_var_decl_local(decl),
            crate::ast::Declaration::Fun(decl) => {
                if decl.body.is_some() {
                    return Err(Error::BlockScopeFunWithBody(decl.name.clone()));
                }
                self.check_fun_decl(decl)
            }
            crate::ast::Declaration::Struct(decl) => self.check_struct_decl(decl),
            crate::ast::Declaration::Union(decl) => self.check_union_decl(decl),
        }
    }

    fn check_var_decl_file(&mut self, decl: &crate::ast::VarDecl) -> Result<(), Error> {
        let crate::ast::VarDecl {
            ident,
            init,
            storage_class,
            ty,
        } = decl;

        self.validate_var_type(ty, storage_class == &Some(crate::ast::StorageClass::Extern))
            .map_err(|_| Error::IncompatibleTypes(ident.span.clone()))?;
        if ty == &ast::VarType::Void {
            return Err(Error::IncompatibleTypes(ident.span.clone()));
        }

        let mut init = match init {
            Some(init) => InitialValue::Initial(self.static_init_from_initializer(ty, init)?),
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
            Some(Attr::Fun { .. } | Attr::Struct { .. } | Attr::Union(..)) => {
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
                    init = old_init.clone();
                } else if !matches!(init, InitialValue::Initial(_))
                    && matches!(old_init, InitialValue::Tentative)
                {
                    init = InitialValue::Tentative;
                }
            }
            Some(Attr::Local(_)) | Some(Attr::Constant { .. }) => {
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

        self.validate_var_type(ty, storage_class == &Some(crate::ast::StorageClass::Extern))
            .map_err(|_| Error::IncompatibleTypes(ident.span.clone()))?;
        if ty == &ast::VarType::Void {
            return Err(Error::IncompatibleTypes(ident.span.clone()));
        }

        match storage_class {
            Some(crate::ast::StorageClass::Extern) => {
                if init.is_some() {
                    return Err(Error::BadInitializer(ident.clone()));
                }
                match self.sym_table.entry(ident.data.clone()) {
                    Entry::Occupied(o) => match o.get() {
                        Attr::Fun { .. } | Attr::Struct(_) | Attr::Union(_) => {
                            return Err(Error::IncompatibleTypes(ident.span.clone()));
                        }
                        Attr::Local(ty0) | Attr::Static { ty: ty0, .. } => {
                            if ty0 != ty {
                                return Err(Error::IncompatibleTypes(ident.span.clone()));
                            }
                        }
                        Attr::Constant { .. } => unreachable!(),
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
                    Some(init) => {
                        InitialValue::Initial(self.static_init_from_initializer(ty, init)?)
                    }
                    None => InitialValue::Initial(vec![StaticInit::Zero(self.sym_table.size(ty))]),
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
                if let Some(init) = init {
                    self.check_init(ty, init)?;
                }
            }
        }

        Ok(())
    }

    fn check_init(
        &mut self,
        target: &ast::VarType,
        init: &mut ast::Initializer,
    ) -> Result<(), Error> {
        let span = init.may_token_span();
        match (target, init) {
            (ast::VarType::Array { element, size }, ast::Initializer::CompoundInit(list)) => {
                if list.len() > *size {
                    return Err(Error::IncompatibleTypes(span.unwrap()));
                }

                for init in list.iter_mut() {
                    self.check_init(element, init)?;
                }

                for _ in list.len()..*size {
                    list.push(self.sym_table.zero_init(element));
                }

                Ok(())
            }
            (
                ast::VarType::Array { element, size },
                ast::Initializer::SingleInit(Expression::String(s, ty)),
            ) => {
                if !element.is_character() {
                    return Err(Error::IncompatibleTypes(span.unwrap()));
                }
                if s.data.len() > *size {
                    return Err(Error::IncompatibleTypes(span.unwrap()));
                }

                *ty = target.clone();
                Ok(())
            }
            (_, ast::Initializer::SingleInit(e)) => {
                if target.is_array() {
                    return Err(Error::IncompatibleTypes(e.token_span()));
                }
                self.check_expression_and_convert(e)?;
                convert_by_assignment(e, target)?;
                Ok(())
            }
            (ast::VarType::Struct(tag), ast::Initializer::CompoundInit(list)) => {
                let StructDef { members, .. } = self.sym_table.struct_def(tag);
                if list.len() > members.len() {
                    return Err(Error::IncompatibleTypes(span.unwrap()));
                }
                let members = members.clone();

                for (init, member) in list.iter_mut().zip(members.iter()) {
                    self.check_init(&member.ty, init)?;
                }

                for m in members.iter().skip(list.len()) {
                    list.push(self.sym_table.zero_init(&m.ty));
                }

                Ok(())
            }
            (ast::VarType::Union(tag), ast::Initializer::CompoundInit(list)) => {
                let UnionDef { members, .. } = self.sym_table.union_def(tag);

                if list.len() > 1 {
                    return Err(Error::IncompatibleTypes(span.unwrap()));
                }

                let ty = members[0].ty.clone();
                self.check_init(&ty, &mut list[0])
            }

            _ => Err(Error::IncompatibleTypes(span.unwrap())),
        }
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
                Some(Attr::Constant { .. } | Attr::Struct { .. } | Attr::Union(_)) => {
                    unreachable!()
                }
                None => Err(Error::IncompatibleTypes(name.span.clone())),
            },
            crate::ast::Expression::Constant(_) => Ok(exp.ty().clone()),
            crate::ast::Expression::Unary { op, exp, ty } => {
                match op.data {
                    ast::UnaryOp::Not => {
                        if !self.check_expression_and_convert(exp)?.is_scalar() {
                            return Err(Error::IncompatibleTypes(exp.token_span()));
                        }
                        *ty = ast::BaseType::Int.into();
                    }
                    ast::UnaryOp::Complement => {
                        *ty = self.check_expression(exp)?;
                        if *ty == ast::BaseType::Double.into() || ty.is_pointer() || ty.is_struct()
                        {
                            return Err(Error::IncompatibleTypes(exp.token_span()));
                        }
                        if ty.is_character() {
                            *ty = ast::BaseType::Int.into();
                            convert_to(exp, &ast::VarType::Base(ast::BaseType::Int));
                        }
                    }
                    ast::UnaryOp::Negate => {
                        *ty = self.check_expression(exp)?;
                        if ty.is_pointer() || !ty.is_scalar() {
                            return Err(Error::IncompatibleTypes(exp.token_span()));
                        }
                        if ty.is_character() {
                            *ty = ast::BaseType::Int.into();
                            convert_to(exp, &ast::VarType::Base(ast::BaseType::Int));
                        }
                    }
                }
                Ok(ty.clone())
            }
            crate::ast::Expression::Binary {
                op,
                lhs,
                rhs,
                ty,
                assign,
            } => {
                let tyl = self.check_expression_and_convert(lhs)?;
                let tyr = self.check_expression_and_convert(rhs)?;

                if *assign && !lhs.is_lvalue() {
                    return Err(Error::IncompatibleTypes(lhs.token_span()));
                }

                match op {
                    ast::BinaryOp::And | ast::BinaryOp::Or => {
                        if !tyl.is_scalar() || !tyr.is_scalar() {
                            return Err(Error::IncompatibleTypes(exp.token_span()));
                        }
                    }
                    ast::BinaryOp::Equal | ast::BinaryOp::NotEqual => {
                        let cty = if tyl.is_pointer() || tyr.is_pointer() {
                            if let Some(cty) = common_pointer_type(lhs, rhs) {
                                cty.clone()
                            } else {
                                return Err(Error::IncompatibleTypes(exp.token_span()));
                            }
                        } else if let (ast::VarType::Base(tyl), ast::VarType::Base(tyr)) =
                            (tyl, tyr)
                        {
                            common_base_type(tyl, tyr).into()
                        } else {
                            return Err(Error::IncompatibleTypes(exp.token_span()));
                        };

                        // convert_to(lhs, &cty);
                        convert_to(rhs, &cty);

                        *ty = ast::BaseType::Int.into();
                    }
                    ast::BinaryOp::Add => {
                        if let (ast::VarType::Base(tyl), ast::VarType::Base(tyr)) = (&tyl, &tyr) {
                            let cty = common_base_type(*tyl, *tyr).into();
                            // convert_to(lhs, &cty);
                            convert_to(rhs, &cty);
                            *ty = cty;
                        } else if self.sym_table.is_pointer_to_complete(&tyl) && tyr.is_integer() {
                            convert_to(rhs, &VarType::Base(ast::BaseType::Long));
                            *ty = tyl.clone();
                        } else if tyl.is_integer() && self.sym_table.is_pointer_to_complete(&tyr) {
                            convert_to(lhs, &VarType::Base(ast::BaseType::Long));
                            *ty = tyr.clone();
                        } else {
                            return Err(Error::IncompatibleTypes(exp.token_span()));
                        }
                    }
                    ast::BinaryOp::Subtract => {
                        if let (ast::VarType::Base(tyl), ast::VarType::Base(tyr)) = (&tyl, &tyr) {
                            let cty = common_base_type(*tyl, *tyr).into();
                            // convert_to(lhs, &cty);
                            convert_to(rhs, &cty);
                            *ty = cty;
                        } else if self.sym_table.is_pointer_to_complete(&tyl) && tyr.is_integer() {
                            convert_to(rhs, &VarType::Base(ast::BaseType::Long));
                            *ty = tyl.clone();
                        } else if self.sym_table.is_pointer_to_complete(&tyl)
                            && self.sym_table.is_pointer_to_complete(&tyr)
                            && tyl == tyr
                        {
                            if *assign {
                                return Err(Error::IncompatibleTypes(exp.token_span()));
                            }
                            *ty = ast::BaseType::Long.into();
                        } else {
                            return Err(Error::IncompatibleTypes(exp.token_span()));
                        }
                    }
                    ast::BinaryOp::ShiftLeft | ast::BinaryOp::ShiftRight => {
                        if let (ast::VarType::Base(tyl), ast::VarType::Base(tyr)) = (&tyl, &tyr) {
                            let cty = ast::VarType::Base(match tyl {
                                ast::BaseType::Char
                                | ast::BaseType::SChar
                                | ast::BaseType::UChar => ast::BaseType::Int,
                                _ => *tyl,
                            });
                            if *tyl == ast::BaseType::Double || *tyr == ast::BaseType::Double {
                                return Err(Error::IncompatibleTypes(exp.token_span()));
                            }
                            // convert_to(lhs, &cty);
                            convert_to(rhs, &cty);
                            *ty = cty;
                        } else {
                            return Err(Error::IncompatibleTypes(exp.token_span()));
                        }
                    }
                    ast::BinaryOp::BitAnd | ast::BinaryOp::BitOr | ast::BinaryOp::BitXor => {
                        if let (ast::VarType::Base(tyl), ast::VarType::Base(tyr)) = (tyl, tyr) {
                            if tyl == BaseType::Double || tyr == BaseType::Double {
                                return Err(Error::IncompatibleTypes(exp.token_span()));
                            }
                            let cty = common_base_type(tyl, tyr).into();
                            // convert_to(lhs, &cty);
                            convert_to(rhs, &cty);
                            *ty = cty;
                        } else {
                            return Err(Error::IncompatibleTypes(exp.token_span()));
                        }
                    }
                    _ => match op {
                        ast::BinaryOp::Multiply | ast::BinaryOp::Divide => {
                            if let (ast::VarType::Base(tyl), ast::VarType::Base(tyr)) = (tyl, tyr) {
                                let cty = common_base_type(tyl, tyr).into();
                                // convert_to(lhs, &cty);
                                convert_to(rhs, &cty);
                                *ty = cty;
                            } else {
                                return Err(Error::IncompatibleTypes(exp.token_span()));
                            }
                        }
                        ast::BinaryOp::Remainder => {
                            if let (ast::VarType::Base(tyl), ast::VarType::Base(tyr)) = (tyl, tyr) {
                                let cty = common_base_type(tyl, tyr);
                                if cty == ast::BaseType::Double {
                                    return Err(Error::IncompatibleTypes(exp.token_span()));
                                }
                                let cty = cty.into();
                                // convert_to(lhs, &cty);
                                convert_to(rhs, &cty);
                                *ty = cty;
                            } else {
                                return Err(Error::IncompatibleTypes(exp.token_span()));
                            }
                        }
                        _ => {
                            if (tyl != tyr) && (tyl.is_pointer() || tyr.is_pointer()) {
                                return Err(Error::IncompatibleTypes(exp.token_span()));
                            }

                            if !self.sym_table.is_complete(&tyl)
                                || !self.sym_table.is_complete(&tyr)
                            {
                                return Err(Error::IncompatibleTypes(exp.token_span()));
                            }

                            if let (ast::VarType::Base(tyl), ast::VarType::Base(tyr)) = (tyl, tyr) {
                                let cty = common_base_type(tyl, tyr).into();
                                // convert_to(lhs, &cty);
                                convert_to(rhs, &cty);
                            }

                            *ty = ast::BaseType::Int.into();
                        }
                    },
                }

                Ok(ty.clone())
            }
            crate::ast::Expression::Assignment { lhs, rhs } => {
                let tyl = self.check_expression_and_convert(lhs)?;
                if !lhs.is_lvalue() {
                    return Err(Error::IncompatibleTypes(lhs.token_span()));
                }
                self.check_expression_and_convert(rhs)?;
                convert_by_assignment(rhs, &tyl)?;
                Ok(tyl)
            }
            crate::ast::Expression::Conditional {
                condition,
                then_branch,
                else_branch,
            } => {
                let cond_ty = self.check_expression_and_convert(condition)?;
                if !cond_ty.is_scalar() {
                    return Err(Error::IncompatibleTypes(condition.token_span()));
                }
                let tyl = self.check_expression_and_convert(then_branch)?;
                let tyr = self.check_expression_and_convert(else_branch)?;

                let cty = if let (ast::VarType::Base(tyl), ast::VarType::Base(tyr)) = (&tyl, &tyr) {
                    common_base_type(*tyl, *tyr).into()
                } else if tyl.is_pointer() || tyr.is_pointer() {
                    if let Some(cty) = common_pointer_type(then_branch, else_branch) {
                        cty.clone()
                    } else {
                        return Err(Error::IncompatibleTypes(exp.token_span()));
                    }
                } else if tyl.is_struct() || tyl.is_union() || tyr.is_struct() || tyr.is_union() {
                    if tyl == tyr {
                        tyl.clone()
                    } else {
                        return Err(Error::IncompatibleTypes(exp.token_span()));
                    }
                } else if tyl == ast::VarType::Void && tyr == ast::VarType::Void {
                    ast::VarType::Void
                } else {
                    return Err(Error::IncompatibleTypes(exp.token_span()));
                };

                convert_to(then_branch, &cty);
                convert_to(else_branch, &cty);

                Ok(cty)
            }
            crate::ast::Expression::FunctionCall {
                callee,
                args,
                ty: fty,
            } => {
                let ty = self.check_expression_and_convert(callee)?;

                match ty {
                    ast::VarType::Pointer(pty) => {
                        if let ast::Ty::Fun(ty) = pty.as_ref() {
                            if (!ty.variable_length_params && ty.params.len() != args.len())
                                || (ty.variable_length_params && ty.params.len() > args.len())
                            {
                                return Err(Error::IncompatibleTypes(callee.token_span()));
                            }
                            let ret = ty.ret.clone();

                            for (arg, ty) in args.iter_mut().zip(ty.params.clone().into_iter()) {
                                self.check_expression_and_convert(arg)?;
                                convert_by_assignment(arg, &ty)?;
                            }
                            *fty = ret.clone();

                            if ret != ast::VarType::Void && !self.sym_table.is_complete(&ret) {
                                return Err(Error::IncompatibleTypes(callee.token_span()));
                            }
                            Ok(ret.clone())
                        } else {
                            Err(Error::IncompatibleTypes(callee.token_span()))
                        }
                    }
                    _ => Err(Error::IncompatibleTypes(callee.token_span())),
                }
            }

            crate::ast::Expression::Cast { target, exp } => {
                self.validate_var_type(target, false)
                    .map_err(|_| Error::IncompatibleTypes(exp.token_span()))?;
                let ty = self.check_expression_and_convert(exp)?;

                if (target.is_pointer() && ty == ast::BaseType::Double.into())
                    || (ty.is_pointer() && target == &VarType::Base(ast::BaseType::Double))
                {
                    return Err(Error::IncompatibleTypes(exp.token_span()));
                }

                if target == &VarType::Void {
                    Ok(target.clone())
                } else if !target.is_scalar() || !ty.is_scalar() {
                    Err(Error::IncompatibleTypes(exp.token_span()))
                } else {
                    Ok(target.clone())
                }
            }
            crate::ast::Expression::Dereference(exp) => {
                let ty = self.check_expression_and_convert(exp)?;
                if let ast::VarType::Pointer(ty) = ty {
                    match ty.as_ref() {
                        ast::Ty::Fun(_) => Err(Error::IncompatibleTypes(exp.token_span())),
                        ast::Ty::Var(ast::VarType::Void) => {
                            Err(Error::IncompatibleTypes(exp.token_span()))
                        }
                        ast::Ty::Var(ty) => Ok(ty.clone()),
                    }
                } else {
                    Err(Error::IncompatibleTypes(exp.token_span()))
                }
            }
            ast::Expression::AddrOf { exp, ty } => {
                if !exp.is_lvalue() {
                    return Err(Error::IncompatibleTypes(exp.token_span()));
                }

                let exp_ty = self.check_expression(exp)?;
                *ty = ast::VarType::Pointer(Box::new(ast::Ty::Var(exp_ty.clone())));
                Ok(ty.clone())
            }
            ast::Expression::Subscript { array, index, ty } => {
                let array_ty = self.check_expression_and_convert(array)?;
                let index_ty = self.check_expression_and_convert(index)?;

                if array_ty.is_pointer() && index_ty.is_integer() {
                    convert_to(index, &ast::VarType::Base(ast::BaseType::Long));
                    *ty = match array_ty {
                        ast::VarType::Pointer(ty) => {
                            if let ast::Ty::Var(ty) = ty.as_ref() {
                                if !self.sym_table.is_complete(ty) {
                                    return Err(Error::IncompatibleTypes(exp.token_span()));
                                }
                                ty.clone()
                            } else {
                                return Err(Error::IncompatibleTypes(exp.token_span()));
                            }
                        }
                        _ => unreachable!(),
                    };
                } else if array_ty.is_integer() && index_ty.is_pointer() {
                    convert_to(array, &ast::BaseType::Long.into());
                    *ty = match index_ty {
                        ast::VarType::Pointer(ty) => {
                            if let ast::Ty::Var(ty) = ty.as_ref() {
                                if !self.sym_table.is_complete(ty) {
                                    return Err(Error::IncompatibleTypes(exp.token_span()));
                                }
                                ty.clone()
                            } else {
                                return Err(Error::IncompatibleTypes(exp.token_span()));
                            }
                        }
                        _ => unreachable!(),
                    };
                } else {
                    return Err(Error::IncompatibleTypes(exp.token_span()));
                }

                Ok(ty.clone())
            }
            ast::Expression::String(_, ty) => Ok(ty.clone()),
            ast::Expression::Sizeof(exp) => {
                let ty = self.check_expression(exp)?;
                if !self.sym_table.is_complete(&ty) {
                    return Err(Error::IncompatibleTypes(exp.token_span()));
                }

                if let ast::VarType::Pointer(ty) = ty {
                    if let ast::Ty::Fun(_) = ty.as_ref() {
                        // I think this is legal but throw an error to pass tests
                        return Err(Error::IncompatibleTypes(exp.token_span()));
                    }
                }

                Ok(ast::BaseType::Ulong.into())
            }
            ast::Expression::SizeofType(ty) => {
                if !self.sym_table.is_complete(&ty.data) {
                    return Err(Error::IncompatibleTypes(exp.token_span()));
                }
                if let ast::VarType::Pointer(ty) = &ty.data {
                    if let ast::Ty::Fun(_) = ty.as_ref() {
                        // I think this is legal but throw an error to pass tests
                        return Err(Error::IncompatibleTypes(exp.token_span()));
                    }
                }
                self.validate_var_type(&ty.data, false)
                    .map_err(|_| Error::IncompatibleTypes(exp.token_span()))?;
                Ok(ast::BaseType::Ulong.into())
            }
            ast::Expression::Dot {
                structure,
                member,
                ty,
            } => {
                let structure_ty = self.check_expression_and_convert(structure)?;
                match structure_ty {
                    ast::VarType::Struct(s) => {
                        let StructDef { members, .. } = self.sym_table.struct_def(&s);
                        if let Some(member) = members.iter().find(|name| name.name == member.data) {
                            *ty = member.ty.clone();
                            Ok(ty.clone())
                        } else {
                            Err(Error::IncompatibleTypes(exp.token_span()))
                        }
                    }
                    ast::VarType::Union(u) => {
                        let UnionDef { members, .. } = self.sym_table.union_def(&u);
                        if let Some(member) = members.iter().find(|name| name.name == member.data) {
                            *ty = member.ty.clone();
                            Ok(ty.clone())
                        } else {
                            Err(Error::IncompatibleTypes(exp.token_span()))
                        }
                    }
                    _ => Err(Error::IncompatibleTypes(exp.token_span())),
                }
            }
            ast::Expression::Arrow {
                pointer,
                member,
                ty,
            } => {
                if let VarType::Pointer(box_ty) = self.check_expression_and_convert(pointer)? {
                    let s = if let Ty::Var(s) = box_ty.as_ref() {
                        s
                    } else {
                        return Err(Error::IncompatibleTypes(exp.token_span()));
                    };
                    match s {
                        ast::VarType::Struct(s) => {
                            let StructDef { members, .. } = self.sym_table.struct_def(s);
                            if let Some(member) =
                                members.iter().find(|name| name.name == member.data)
                            {
                                *ty = member.ty.clone();
                                Ok(ty.clone())
                            } else {
                                Err(Error::IncompatibleTypes(exp.token_span()))
                            }
                        }

                        ast::VarType::Union(u) => {
                            let UnionDef { members, .. } = self.sym_table.union_def(u);
                            if let Some(member) =
                                members.iter().find(|name| name.name == member.data)
                            {
                                *ty = member.ty.clone();
                                Ok(ty.clone())
                            } else {
                                Err(Error::IncompatibleTypes(exp.token_span()))
                            }
                        }
                        _ => Err(Error::IncompatibleTypes(exp.token_span())),
                    }
                } else {
                    Err(Error::IncompatibleTypes(exp.token_span()))
                }
            }
            ast::Expression::Increment { exp, .. } | ast::Expression::Decrement { exp, .. } => {
                if let Expression::Var(name, _) = exp.as_ref() {
                    if let Some(Attr::Fun { .. }) = self.sym_table.get(&name.data) {
                        return Err(Error::IncompatibleTypes(exp.token_span()));
                    }
                }

                let ty = self.check_expression_and_convert(exp)?;
                if !ty.is_scalar() {
                    return Err(Error::IncompatibleTypes(exp.token_span()));
                }
                if !exp.is_lvalue() {
                    return Err(Error::IncompatibleTypes(exp.token_span()));
                }

                Ok(ty)
            }
        }
    }

    fn check_expression_and_convert(
        &mut self,
        exp: &mut crate::ast::Expression,
    ) -> Result<ast::VarType, Error> {
        match self.check_expression(exp)? {
            VarType::Array { element, .. } => {
                let ty = VarType::Pointer(Box::new(ast::Ty::Var(*element)));
                *exp = Expression::AddrOf {
                    exp: Box::new(exp.clone()),
                    ty: ty.clone(),
                };
                Ok(ty)
            }
            VarType::Struct(s) => {
                if self.sym_table.is_complete(&VarType::Struct(s.clone())) {
                    Ok(VarType::Struct(s))
                } else {
                    Err(Error::IncompatibleTypes(exp.token_span()))
                }
            }
            VarType::Union(u) => {
                if self.sym_table.is_complete(&VarType::Union(u.clone())) {
                    Ok(VarType::Union(u))
                } else {
                    Err(Error::IncompatibleTypes(exp.token_span()))
                }
            }
            t => Ok(t),
        }
    }

    fn check_statement(
        &mut self,
        stmt: &mut crate::ast::Statement,
        ret_type: &ast::VarType,
    ) -> Result<(), Error> {
        match stmt {
            crate::ast::Statement::Return(exp) => match (ret_type, exp) {
                (ast::VarType::Void, Some(exp)) => Err(Error::IncompatibleTypes(exp.token_span())),
                (ast::VarType::Void, None) => Ok(()),
                (ret_type, Some(exp)) => {
                    self.check_expression_and_convert(exp)?;
                    convert_by_assignment(exp, ret_type)?;
                    Ok(())
                }
                // todo span
                _ => Err(Error::IncompatibleTypes(0..0)),
            },
            crate::ast::Statement::Expression(exp) => {
                self.check_expression_and_convert(exp)?;
                Ok(())
            }
            crate::ast::Statement::If {
                condition,
                then_branch,
                else_branch,
            } => {
                let cond_ty = self.check_expression_and_convert(condition)?;

                if !cond_ty.is_scalar() {
                    return Err(Error::IncompatibleTypes(condition.token_span()));
                }

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
                let cond_ty = self.check_expression_and_convert(condition)?;
                if !cond_ty.is_scalar() {
                    return Err(Error::IncompatibleTypes(condition.token_span()));
                }
                self.check_statement(body, ret_type)?;
                Ok(())
            }
            crate::ast::Statement::DoWhile {
                label: _,
                condition,
                body,
            } => {
                self.check_statement(body, ret_type)?;
                let cond_ty = self.check_expression_and_convert(condition)?;
                if !cond_ty.is_scalar() {
                    return Err(Error::IncompatibleTypes(condition.token_span()));
                }
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
                            self.check_expression_and_convert(exp)?;
                        }
                    }
                }
                if let Some(condition) = condition {
                    let cond_ty = self.check_expression_and_convert(condition)?;
                    if !cond_ty.is_scalar() {
                        return Err(Error::IncompatibleTypes(condition.token_span()));
                    }
                }
                if let Some(step) = step {
                    self.check_expression_and_convert(step)?;
                }
                self.check_statement(body, ret_type)?;
                Ok(())
            }
            crate::ast::Statement::Null => Ok(()),
            crate::ast::Statement::Goto(_) => Ok(()),
            crate::ast::Statement::Label {
                statement: stmt, ..
            } => self.check_statement(stmt, ret_type),
            crate::ast::Statement::Case { exp, statement, .. } => {
                let ty = self.check_expression_and_convert(exp)?;

                if !ty.is_integer() || !matches!(exp, Expression::Constant(_)) {
                    return Err(Error::IncompatibleTypes(exp.token_span()));
                }

                self.check_statement(statement, ret_type)?;
                Ok(())
            }
            crate::ast::Statement::Default { statement, .. } => {
                self.check_statement(statement, ret_type)?;
                Ok(())
            }
            crate::ast::Statement::Switch {
                exp,
                statement,
                labels,
                ..
            } => {
                let ty = self.check_expression_and_convert(exp)?;
                if !ty.is_integer() {
                    return Err(Error::IncompatibleTypes(exp.token_span()));
                }
                match &ty {
                    VarType::Base(BaseType::Char | BaseType::SChar) => {
                        convert_to(exp, &ast::BaseType::Int.into());
                    }
                    VarType::Base(BaseType::UChar) => {
                        convert_to(exp, &ast::BaseType::Uint.into());
                    }
                    _ => {}
                }
                let bits = 8 * self.sym_table.size(exp.ty());
                let new_cases = labels
                    .cases
                    .iter()
                    .map(|(k, v)| (*k << (64 - bits) >> (64 - bits), v.clone()))
                    .collect::<BTreeMap<u64, EcoString>>();

                if new_cases.len() != labels.cases.len() {
                    return Err(Error::IncompatibleTypes(exp.token_span()));
                }

                labels.cases = new_cases;

                self.check_statement(statement, ret_type)?;
                Ok(())
            }
        }
    }

    fn check_struct_decl(&mut self, decl: &ast::StructDecl) -> Result<(), Error> {
        if decl.member_decls.is_empty() {
            return Ok(());
        }

        if self.sym_table.contains_key(&decl.tag.data) {
            return Err(Error::Redefined(decl.tag.clone()));
        }

        let mut members = Vec::new();

        let mut struct_size = 0;
        let mut struct_align = 0;

        for member in &decl.member_decls {
            let size = self.sym_table.size(&member.ty);
            let align = if let VarType::Array { element, .. } = &member.ty {
                // HACK
                // TODO: Read SystemV ABI
                self.sym_table.alignment(element)
            } else {
                self.sym_table.alignment(&member.ty)
            };

            let offset = round_up(struct_size, align);
            members.push(StructMember {
                name: member.name.clone(),
                offset,
                ty: member.ty.clone(),
            });

            struct_size = offset + size;
            struct_align = std::cmp::max(struct_align, align);
        }
        struct_size = round_up(struct_size, struct_align);

        self.sym_table.insert(
            decl.tag.data.clone(),
            Attr::Struct(StructDef {
                members,
                size: struct_size,
                alignment: struct_align,
            }),
        );
        self.validate_struct_definition(decl)?;
        Ok(())
    }

    fn check_union_decl(&mut self, decl: &ast::UnionDecl) -> Result<(), Error> {
        if decl.member_decls.is_empty() {
            return Ok(());
        }

        if self.sym_table.contains_key(&decl.tag.data) {
            return Err(Error::Redefined(decl.tag.clone()));
        }

        let mut members = Vec::new();

        let mut union_size = 0;
        let mut union_align = 0;

        for member in &decl.member_decls {
            let size = self.sym_table.size(&member.ty);
            let align = if let VarType::Array { element, .. } = &member.ty {
                // HACK
                // TODO: Read SystemV ABI
                self.sym_table.alignment(element)
            } else {
                self.sym_table.alignment(&member.ty)
            };

            members.push(UnionMember {
                name: member.name.clone(),
                ty: member.ty.clone(),
            });

            union_size = std::cmp::max(union_size, size);
            union_align = std::cmp::max(union_align, align);
        }
        union_size = round_up(union_size, union_align);

        self.sym_table.insert(
            decl.tag.data.clone(),
            Attr::Union(UnionDef {
                members,
                size: union_size,
                alignment: union_align,
            }),
        );
        self.validate_union_definition(decl)?;
        Ok(())
    }

    fn validate_struct_definition(&self, decl: &ast::StructDecl) -> Result<(), Error> {
        let mut member_names = HashSet::new();

        for member in &decl.member_decls {
            if !member_names.insert(member.name.clone()) {
                todo!()
            }

            if self.validate_var_type(&member.ty, false).is_err() {
                todo!()
            }
        }

        Ok(())
    }

    fn validate_union_definition(&self, decl: &ast::UnionDecl) -> Result<(), Error> {
        let mut member_names = HashSet::new();

        for member in &decl.member_decls {
            if !member_names.insert(member.name.clone()) {
                todo!()
            }

            if self.validate_var_type(&member.ty, false).is_err() {
                todo!()
            }
        }

        Ok(())
    }

    fn validate_var_type(
        &self,
        ty: &ast::VarType,
        allow_incomplete_struct: bool,
    ) -> Result<(), ()> {
        match ty {
            VarType::Array { element, .. } => {
                if !self.sym_table.is_complete(element) {
                    return Err(());
                }
                self.validate_var_type(element, allow_incomplete_struct)?;
            }
            VarType::Pointer(ty) => match ty.as_ref() {
                ast::Ty::Fun(ty) => {
                    self.validate_fun_type(ty, allow_incomplete_struct)?;
                }
                ast::Ty::Var(ty) => {
                    self.validate_var_type(ty, true)?;
                }
            },
            VarType::Struct(tag) => {
                if !allow_incomplete_struct
                    && !matches!(self.sym_table.get(tag), Some(Attr::Struct { .. }))
                {
                    return Err(());
                }
            }
            VarType::Union(tag) => {
                if !allow_incomplete_struct
                    && !matches!(self.sym_table.get(tag), Some(Attr::Union { .. }))
                {
                    return Err(());
                }
            }
            _ => {}
        }

        Ok(())
    }

    fn validate_fun_type(
        &self,
        ty: &ast::FunType,
        allow_incomplete_struct: bool,
    ) -> Result<(), ()> {
        for ty in &ty.params {
            if ty == &ast::VarType::Void {
                return Err(());
            }
            self.validate_var_type(ty, allow_incomplete_struct)?;
        }

        if matches!(ty.ret, ast::VarType::Array { .. }) {
            return Err(());
        }

        self.validate_var_type(&ty.ret, allow_incomplete_struct)?;

        Ok(())
    }
}

impl Display for StaticInit {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self {
            StaticInit::Int(x) => writeln!(f, ".long {}", x)?,
            StaticInit::Uint(x) => writeln!(f, ".long {}", x)?,
            StaticInit::Long(x) => writeln!(f, ".quad {}", x)?,
            StaticInit::Ulong(x) => writeln!(f, ".quad {}", x)?,
            StaticInit::Double(d) => {
                writeln!(f, ".quad {}", d.to_bits())?;
                writeln!(f, "# {:+e}", d)?;
            }
            StaticInit::Zero(size) => writeln!(f, ".zero {}", size)?,
            StaticInit::Char(c) => writeln!(f, ".byte {}", c)?,
            StaticInit::UChar(c) => writeln!(f, ".byte {}", c)?,
            StaticInit::Pointer(name) => writeln!(f, ".quad {}", name)?,
            StaticInit::String { data, pad } => {
                if data.iter().all(|&c| c.is_ascii()) {
                    write!(f, ".ascii \"")?;
                    for c in data {
                        write!(f, "{}", std::ascii::escape_default(*c))?;
                    }
                    writeln!(f, "\"")?;
                } else {
                    for c in data {
                        writeln!(f, ".byte {}", c)?;
                    }
                }

                if *pad > 0 {
                    writeln!(f, ".zero {}", pad)?;
                }
            }
        }

        Ok(())
    }
}
