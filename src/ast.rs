use ecow::EcoString;

use crate::{
    lexer::{Constant, Suffix, Token},
    semantics::type_check::StaticInit,
    span::{HasSpan, MayHasSpan, Spanned},
};

#[derive(Debug)]
pub struct Program {
    pub decls: Vec<Declaration>,
}

#[derive(Debug)]
pub enum Declaration {
    VarDecl(VarDecl),
    FunDecl(FunDecl),
    StructDecl(StructDecl),
}

#[derive(Debug)]
pub struct VarDecl {
    pub ident: Spanned<EcoString>,
    pub init: Option<Initializer>,
    pub ty: VarType,
    pub storage_class: Option<StorageClass>,
}

#[derive(Debug)]
pub struct FunDecl {
    pub name: Spanned<EcoString>,
    pub params: Vec<Spanned<EcoString>>,
    pub body: Option<Block>,
    pub ty: FunType,
    pub storage_class: Option<StorageClass>,
}

#[derive(Debug)]
pub struct StructDecl {
    pub tag: Spanned<EcoString>,
    pub member_decls: Vec<MemberDecl>,
}

#[derive(Debug)]
pub struct MemberDecl {
    pub name: EcoString,
    pub ty: VarType,
}

#[derive(Debug, Clone)]
pub enum Initializer {
    SingleInit(Expression),
    CompoundInit(Vec<Initializer>),
}

impl Initializer {
    pub fn zero_base(ty: BaseType) -> Self {
        match ty {
            BaseType::Char => {
                Self::SingleInit(Expression::Constant(Spanned::new_null(Const::Char(0))))
            }
            BaseType::SChar => {
                Self::SingleInit(Expression::Constant(Spanned::new_null(Const::Char(0))))
            }
            BaseType::UChar => {
                Self::SingleInit(Expression::Constant(Spanned::new_null(Const::UChar(0))))
            }
            BaseType::Int => {
                Self::SingleInit(Expression::Constant(Spanned::new_null(Const::Int(0))))
            }
            BaseType::Uint => {
                Self::SingleInit(Expression::Constant(Spanned::new_null(Const::Uint(0))))
            }
            BaseType::Long => {
                Self::SingleInit(Expression::Constant(Spanned::new_null(Const::Long(0))))
            }
            BaseType::Ulong => {
                Self::SingleInit(Expression::Constant(Spanned::new_null(Const::Ulong(0))))
            }
            BaseType::Double => {
                Self::SingleInit(Expression::Constant(Spanned::new_null(Const::Double(0.0))))
            }
        }
    }
}

impl MayHasSpan for Initializer {
    fn may_span(&self) -> Option<std::ops::Range<usize>> {
        match self {
            Self::SingleInit(exp) => Some(exp.span()),
            Self::CompoundInit(inits) => {
                let start = inits.first()?.may_span()?.start;
                let end = inits.last()?.may_span()?.end;
                Some(start..end)
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
pub enum StorageClass {
    Static,
    Extern,
}

#[derive(Debug)]
pub struct Block(pub Vec<BlockItem>);

#[derive(Debug)]
pub enum BlockItem {
    Declaration(Declaration),
    Statement(Statement),
}

#[derive(Debug)]
pub enum ForInit {
    VarDecl(VarDecl),
    Expression(Expression),
}

#[derive(Debug)]
pub enum Statement {
    Return(Option<Expression>),
    Expression(Expression),
    If {
        condition: Expression,
        then_branch: Box<Statement>,
        else_branch: Option<Box<Statement>>,
    },
    Compound(Block),
    Break {
        label: EcoString,
        span: std::ops::Range<usize>,
    },
    Continue {
        label: EcoString,
        span: std::ops::Range<usize>,
    },
    While {
        label: EcoString,
        condition: Expression,
        body: Box<Statement>,
    },
    DoWhile {
        label: EcoString,
        condition: Expression,
        body: Box<Statement>,
    },
    For {
        label: EcoString,
        init: Option<ForInit>,
        condition: Option<Expression>,
        step: Option<Expression>,
        body: Box<Statement>,
    },
    Null,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Const {
    Char(i8),
    UChar(u8),
    Int(i32),
    Long(i64),
    Uint(u32),
    Ulong(u64),
    Double(f64),
}

impl Const {
    pub fn get_int(&self) -> i32 {
        match self {
            Self::Char(i) => *i as i32,
            Self::UChar(i) => *i as i32,
            Self::Int(i) => *i,
            Self::Uint(i) => *i as i32,
            Self::Long(i) => *i as i32,
            Self::Ulong(i) => *i as i32,
            Self::Double(i) => *i as i32,
        }
    }
    pub fn get_uint(&self) -> u32 {
        match self {
            Self::Char(i) => *i as u32,
            Self::UChar(i) => *i as u32,
            Self::Int(i) => *i as u32,
            Self::Uint(i) => *i,
            Self::Long(i) => *i as u32,
            Self::Ulong(i) => *i as u32,
            Self::Double(i) => *i as u32,
        }
    }
    pub fn get_long(&self) -> i64 {
        match self {
            Self::Char(i) => *i as i64,
            Self::UChar(i) => *i as i64,
            Self::Int(i) => *i as i64,
            Self::Uint(i) => *i as i64,
            Self::Long(i) => *i,
            Self::Ulong(i) => *i as i64,
            Self::Double(i) => *i as i64,
        }
    }
    pub fn get_ulong(&self) -> u64 {
        match self {
            Self::Char(i) => *i as u64,
            Self::UChar(i) => *i as u64,
            Self::Int(i) => *i as u64,
            Self::Uint(i) => *i as u64,
            Self::Long(i) => *i as u64,
            Self::Ulong(i) => *i,
            Self::Double(i) => *i as u64,
        }
    }
    pub fn get_double(&self) -> f64 {
        match self {
            Self::Char(i) => *i as f64,
            Self::UChar(i) => *i as f64,
            Self::Int(i) => *i as f64,
            Self::Uint(i) => *i as f64,
            Self::Long(i) => *i as f64,
            Self::Ulong(i) => *i as f64,
            Self::Double(i) => *i,
        }
    }

    pub fn get_static_init(&self, ty: &VarType) -> Option<StaticInit> {
        match ty {
            VarType::Void => None,
            VarType::Base(base) => match base {
                BaseType::Char => Some(StaticInit::Char(self.get_int() as i8)),
                BaseType::SChar => Some(StaticInit::Char(self.get_int() as i8)),
                BaseType::UChar => Some(StaticInit::UChar(self.get_uint() as u8)),
                BaseType::Int => Some(StaticInit::Int(self.get_int())),
                BaseType::Uint => Some(StaticInit::Uint(self.get_uint())),
                BaseType::Long => Some(StaticInit::Long(self.get_long())),
                BaseType::Ulong => Some(StaticInit::Ulong(self.get_ulong())),
                BaseType::Double => Some(StaticInit::Double(self.get_double())),
            },
            VarType::Pointer(_) => match self {
                Self::Int(0) => Some(StaticInit::Ulong(0)),
                Self::Uint(0) => Some(StaticInit::Ulong(0)),
                Self::Long(0) => Some(StaticInit::Ulong(0)),
                Self::Ulong(0) => Some(StaticInit::Ulong(0)),
                _ => None,
            },
            VarType::Array { .. } => None,
            VarType::Struct(_) => None,
        }
    }
}

#[derive(Debug, Clone)]
pub enum Expression {
    Var(Spanned<EcoString>, VarType),
    Cast {
        target: VarType,
        exp: Box<Expression>,
    },
    Constant(Spanned<Const>),
    Unary {
        op: Spanned<UnaryOp>,
        exp: Box<Expression>,
        ty: VarType,
    },
    Binary {
        op: BinaryOp,
        lhs: Box<Expression>,
        rhs: Box<Expression>,
        ty: VarType,
    },
    Assignment {
        lhs: Box<Expression>,
        rhs: Box<Expression>,
    },
    Conditional {
        condition: Box<Expression>,
        then_branch: Box<Expression>,
        else_branch: Box<Expression>,
    },
    FunctionCall {
        name: Spanned<EcoString>,
        args: Vec<Expression>,
        ty: VarType,
    },
    Dereference(Box<Expression>),
    AddrOf {
        exp: Box<Expression>,
        ty: VarType,
    },
    Subscript {
        array: Box<Expression>,
        index: Box<Expression>,
        ty: VarType,
    },
    String(Spanned<Vec<u8>>, VarType),
    Sizeof(Box<Expression>),
    SizeofType(Spanned<VarType>),
    Dot {
        structure: Box<Expression>,
        member: Spanned<EcoString>,
        ty: VarType,
    },
    Arrow {
        pointer: Box<Expression>,
        member: Spanned<EcoString>,
        ty: VarType,
    },
}

impl Expression {
    pub fn ty(&self) -> &VarType {
        match self {
            Self::Var(_, ty) => ty,
            Self::Cast { target, .. } => target,
            Self::Constant(Spanned { data, .. }) => match data {
                Const::Int(_) => &VarType::Base(BaseType::Int),
                Const::Long(_) => &VarType::Base(BaseType::Long),
                Const::Uint(_) => &VarType::Base(BaseType::Uint),
                Const::Ulong(_) => &VarType::Base(BaseType::Ulong),
                Const::Double(_) => &VarType::Base(BaseType::Double),
                Const::Char(_) => &VarType::Base(BaseType::Char),
                Const::UChar(_) => &VarType::Base(BaseType::UChar),
            },
            Self::Unary { ty, .. } => ty,
            Self::Binary { ty, .. } => ty,
            Self::Assignment { lhs, .. } => lhs.ty(),
            Self::Conditional { then_branch, .. } => then_branch.ty(),
            Self::FunctionCall { ty, .. } => ty,
            Self::Dereference(exp) => match exp.ty() {
                VarType::Pointer(ty) => match ty.as_ref() {
                    Ty::Var(ty) => ty,
                    Ty::Fun(_) => panic!(
                        "Dereference of function pointer. This should be caught by type checker."
                    ),
                },
                _ => panic!("Dereference of non-pointer. This should be caught by type checker."),
            },
            Self::AddrOf { ty, .. } => ty,
            Self::Subscript { ty, .. } => ty,
            Self::String(_, ty) => ty,
            Self::Sizeof(_) => &VarType::Base(BaseType::Ulong),
            Self::SizeofType(_) => &VarType::Base(BaseType::Ulong),
            Self::Dot { ty, .. } => ty,
            Self::Arrow { ty, .. } => ty,
        }
    }

    pub fn is_null_pointer_constant(&self) -> bool {
        // TODO: const expr
        matches!(
            self,
            Self::Constant(Spanned {
                data: Const::Int(0),
                ..
            }) | Self::Constant(Spanned {
                data: Const::Uint(0),
                ..
            }) | Self::Constant(Spanned {
                data: Const::Long(0),
                ..
            }) | Self::Constant(Spanned {
                data: Const::Ulong(0),
                ..
            })
        )
    }

    pub fn is_lvalue(&self) -> bool {
        match self {
            Self::Var(_, _)
            | Self::Dereference(_)
            | Self::Subscript { .. }
            | Self::String(..)
            | Self::Arrow { .. } => true,
            Self::Dot { structure, .. } => structure.is_lvalue(),
            _ => false,
        }
    }
}

impl HasSpan for Expression {
    fn span(&self) -> std::ops::Range<usize> {
        match self {
            Self::Var(ident, ..) => ident.span.clone(),
            Self::Constant(constant) => constant.span.clone(),
            Self::Unary { op, exp, .. } => op.span.start..exp.span().end,
            Self::Binary { lhs, rhs, .. } => lhs.span().start..rhs.span().end,
            Self::Assignment { lhs, rhs, .. } => lhs.span().start..rhs.span().end,
            Self::Conditional {
                condition,
                else_branch,
                ..
            } => condition.span().start..else_branch.span().end,
            Self::FunctionCall { name, .. } => name.span.clone(),
            Self::Cast { exp, .. } => exp.span(),
            Self::Dereference(exp) => exp.span(),
            Self::AddrOf { exp, .. } => exp.span(),
            Self::Subscript { array, index, .. } => array.span().start..index.span().end,
            Self::String(s, _) => s.span.clone(),
            Self::Sizeof(exp) => exp.span(),
            Self::SizeofType(ty) => ty.span.clone(),
            Self::Dot {
                structure, member, ..
            } => structure.span().start..member.span.end,
            Self::Arrow {
                pointer, member, ..
            } => pointer.span().start..member.span.end,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Ty {
    Var(VarType),
    Fun(FunType),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BaseType {
    Char,
    SChar,
    UChar,
    Int,
    Long,
    Uint,
    Ulong,
    Double,
}

impl BaseType {
    pub fn size(&self) -> usize {
        match self {
            Self::Char => 1,
            Self::SChar => 1,
            Self::UChar => 1,
            Self::Int => 4,
            Self::Uint => 4,
            Self::Long => 8,
            Self::Ulong => 8,
            Self::Double => 8,
        }
    }

    pub fn alignment(&self) -> usize {
        self.size()
    }

    pub fn is_integer(&self) -> bool {
        matches!(
            self,
            Self::Int
                | Self::Uint
                | Self::Long
                | Self::Ulong
                | Self::Char
                | Self::SChar
                | Self::UChar
        )
    }

    pub fn is_character(&self) -> bool {
        matches!(self, Self::Char | Self::SChar | Self::UChar)
    }

    pub fn is_signed(&self) -> bool {
        matches!(self, Self::Char | Self::SChar | Self::Int | Self::Long)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VarType {
    Void,
    Base(BaseType),
    Pointer(Box<Ty>),
    Array { element: Box<VarType>, size: usize },
    Struct(EcoString),
}

impl From<BaseType> for VarType {
    fn from(base: BaseType) -> Self {
        Self::Base(base)
    }
}

impl VarType {
    pub fn is_integer(&self) -> bool {
        if let Self::Base(base) = self {
            base.is_integer()
        } else {
            false
        }
    }

    pub fn is_scalar(&self) -> bool {
        matches!(self, Self::Base(_) | Self::Pointer(_))
    }

    pub fn is_signed(&self) -> bool {
        if let Self::Base(base) = self {
            base.is_signed()
        } else {
            false
        }
    }

    pub fn is_pointer(&self) -> bool {
        matches!(self, Self::Pointer(_))
    }

    pub fn is_array(&self) -> bool {
        matches!(self, Self::Array { .. })
    }

    pub fn is_character(&self) -> bool {
        if let Self::Base(base) = self {
            base.is_character()
        } else {
            false
        }
    }

    pub fn is_struct(&self) -> bool {
        matches!(self, Self::Struct(_))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunType {
    pub params: Vec<VarType>,
    pub ret: VarType,
}

#[derive(Debug, Clone, Copy)]
pub enum UnaryOp {
    Complement,
    Negate,
    Not,
}

impl TryFrom<&Token> for UnaryOp {
    type Error = ();

    fn try_from(token: &Token) -> Result<Self, Self::Error> {
        match token {
            Token::Tilde => Ok(Self::Complement),
            Token::Hyphen => Ok(Self::Negate),
            Token::Exclamation => Ok(Self::Not),
            _ => Err(()),
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub enum BinaryOp {
    Add,
    Subtract,
    Multiply,
    Divide,
    Remainder,
    And,
    Or,
    Equal,
    NotEqual,
    LessThan,
    LessOrEqual,
    GreaterThan,
    GreaterOrEqual,
}

impl BinaryOp {
    fn precedence(&self) -> usize {
        match self {
            Self::Or => 5,
            Self::And => 10,
            Self::Equal | Self::NotEqual => 30,
            Self::LessThan | Self::LessOrEqual | Self::GreaterThan | Self::GreaterOrEqual => 35,
            Self::Add | Self::Subtract => 45,
            Self::Multiply | Self::Divide | Self::Remainder => 50,
        }
    }
}

impl TryFrom<&Token> for BinaryOp {
    type Error = ();

    fn try_from(token: &Token) -> Result<Self, Self::Error> {
        match token {
            Token::Plus => Ok(Self::Add),
            Token::Hyphen => Ok(Self::Subtract),
            Token::Asterisk => Ok(Self::Multiply),
            Token::Slash => Ok(Self::Divide),
            Token::Percent => Ok(Self::Remainder),
            Token::TwoAmpersands => Ok(Self::And),
            Token::TwoPipes => Ok(Self::Or),
            Token::TwoEquals => Ok(Self::Equal),
            Token::ExclamationEquals => Ok(Self::NotEqual),
            Token::LessThan => Ok(Self::LessThan),
            Token::LessThanEquals => Ok(Self::LessOrEqual),
            Token::GreaterThan => Ok(Self::GreaterThan),
            Token::GreaterThanEquals => Ok(Self::GreaterOrEqual),
            _ => Err(()),
        }
    }
}

pub fn parse(tokens: &[Spanned<Token>]) -> Result<Program, Error> {
    let mut parser = Parser { tokens, index: 0 };
    parser.parse_program()
}

struct Parser<'a> {
    tokens: &'a [Spanned<Token>],
    index: usize,
}

#[derive(Debug)]
pub enum ExpectedToken {
    Token(Token),
    Ident,
    Constant,
    Eof,
    Specifier,
    Declarator,
}

enum TypeSpecifier {
    Void,
    Char,
    Int,
    Long,
    Unsigned,
    Signed,
    Double,
    Struct(EcoString),
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Unexpected token: {0:?}, expected {1:?}")]
    Unexpected(Spanned<Token>, ExpectedToken),
    #[error("Unexpected Eof")]
    UnexpectedEof,
    #[error("Malformed expression: {0:?}")]
    MalformedExpression(Spanned<Token>),
    #[error("Malformed body: {0:?}")]
    MalformedBody(Spanned<Token>),
    #[error("Conflicting specifier: {0:?}")]
    ConflictingSpecifier(std::ops::Range<usize>),
    #[error("No type specifier")]
    NoTypeSpecifier(std::ops::Range<usize>),
    #[error("Bad type specifier")]
    BadTypeSpecifier(std::ops::Range<usize>),
    #[error("Unexpected specifier")]
    UnexpectedSpecifier(Spanned<Token>),
    #[error("Function type isn't allowed here")]
    NotVarType(std::ops::Range<usize>),
    #[error("Variable type isn't allowed here")]
    NotFunType(std::ops::Range<usize>),
    #[error("Array length must be a constant integer")]
    BadArrayLength(std::ops::Range<usize>),
}

impl From<Error> for () {
    fn from(_: Error) {}
}

impl MayHasSpan for Error {
    fn may_span(&self) -> Option<std::ops::Range<usize>> {
        match self {
            Error::Unexpected(spanned, _) => Some(spanned.span.clone()),
            Error::UnexpectedEof => None,
            Error::MalformedExpression(spanned) => Some(spanned.span.clone()),
            Error::MalformedBody(spanned) => Some(spanned.span.clone()),
            Error::ConflictingSpecifier(span) => Some(span.clone()),
            Error::NoTypeSpecifier(span) => Some(span.clone()),
            Error::BadTypeSpecifier(span) => Some(span.clone()),
            Error::UnexpectedSpecifier(spanned) => Some(spanned.span.clone()),
            Error::NotVarType(span) => Some(span.clone()),
            Error::NotFunType(span) => Some(span.clone()),
            Error::BadArrayLength(span) => Some(span.clone()),
        }
    }
}

fn solve_type_specifier(ty: &[Spanned<TypeSpecifier>]) -> Result<VarType, Error> {
    debug_assert!(!ty.is_empty());

    let mut int = false;
    let mut long = false;
    let mut char = false;
    let mut signed = false;
    let mut unsigned = false;

    if matches!(
        ty,
        [Spanned {
            data: TypeSpecifier::Double,
            ..
        }]
    ) {
        return Ok(BaseType::Double.into());
    }

    if matches!(
        ty,
        [Spanned {
            data: TypeSpecifier::Void,
            ..
        }]
    ) {
        return Ok(VarType::Void);
    }

    if let [Spanned {
        data: TypeSpecifier::Struct(tag),
        ..
    }] = ty
    {
        return Ok(VarType::Struct(tag.clone()));
    }

    for s in ty {
        match s.data {
            TypeSpecifier::Void => {
                return Err(Error::ConflictingSpecifier(s.span.clone()));
            }
            TypeSpecifier::Char => {
                if int || long || char {
                    return Err(Error::ConflictingSpecifier(s.span.clone()));
                }
                char = true;
            }
            TypeSpecifier::Int => {
                if int || char {
                    return Err(Error::ConflictingSpecifier(s.span.clone()));
                }
                int = true;
            }
            TypeSpecifier::Long => {
                if char || long {
                    return Err(Error::ConflictingSpecifier(s.span.clone()));
                }
                long = true;
            }
            TypeSpecifier::Signed => {
                if signed || unsigned {
                    return Err(Error::ConflictingSpecifier(s.span.clone()));
                }
                signed = true;
            }
            TypeSpecifier::Unsigned => {
                if signed || unsigned {
                    return Err(Error::ConflictingSpecifier(s.span.clone()));
                }
                unsigned = true;
            }
            TypeSpecifier::Double => {
                return Err(Error::BadTypeSpecifier(s.span.clone()));
            }
            TypeSpecifier::Struct(_) => {
                return Err(Error::BadTypeSpecifier(s.span.clone()));
            }
        }
    }

    let base_ty = match (char, int, long) {
        (true, _, _) => BaseType::Char,
        (_, _, true) => BaseType::Long,
        _ => BaseType::Int,
    };

    Ok(match base_ty {
        BaseType::Char => {
            if unsigned {
                BaseType::UChar
            } else if signed {
                BaseType::SChar
            } else {
                BaseType::Char
            }
        }
        BaseType::Long => {
            if unsigned {
                BaseType::Ulong
            } else {
                BaseType::Long
            }
        }
        BaseType::Int => {
            if unsigned {
                BaseType::Uint
            } else {
                BaseType::Int
            }
        }
        _ => unreachable!(),
    }
    .into())
}

#[derive(Debug)]
enum Declarator {
    Ident(EcoString),
    Pointer(Spanned<Box<Declarator>>),
    Array {
        decl: Spanned<Box<Declarator>>,
        size: usize,
    },
    Fun {
        params: Vec<ParamInfo>,
        decl: Spanned<Box<Declarator>>,
    },
}

#[derive(Debug)]
struct ParamInfo {
    ty: VarType,
    decl: Spanned<Declarator>,
}

#[allow(clippy::type_complexity)]
fn process_declarator(
    decl: Spanned<Declarator>,
    base_type: VarType,
) -> Result<(Spanned<EcoString>, Ty, Vec<Spanned<EcoString>>), Error> {
    let span = decl.span.clone();
    match decl.data {
        Declarator::Ident(name) => {
            Ok((Spanned { data: name, span }, Ty::Var(base_type), Vec::new()))
        }
        Declarator::Pointer(d) => {
            let derived_type = VarType::Pointer(Box::new(Ty::Var(base_type)));
            process_declarator(d.map(|d| *d), derived_type)
        }
        Declarator::Fun { params, decl } => {
            let mut param_names = Vec::new();
            let mut param_types = Vec::new();

            for ParamInfo { ty, decl } in params {
                let (name, ty, _) = process_declarator(decl, ty)?;

                match ty {
                    Ty::Fun(_) => return Err(Error::NotVarType(name.span.clone())),
                    Ty::Var(var_ty) => {
                        param_types.push(var_ty);
                    }
                }
                param_names.push(name);
            }
            match *decl.data {
                Declarator::Ident(name) => {
                    let derived_type = Ty::Fun(FunType {
                        params: param_types,
                        ret: base_type,
                    });

                    Ok((Spanned { data: name, span }, derived_type, param_names))
                }
                Declarator::Pointer(decl) => {
                    let fun_ptr = VarType::Pointer(Box::new(Ty::Fun(FunType {
                        params: param_types,
                        ret: base_type,
                    })));

                    let (name, ty, _) = process_declarator(decl.map(|d| *d), fun_ptr)?;
                    Ok((name, ty, param_names))
                }
                Declarator::Fun { .. } => Err(Error::NotVarType(decl.span.clone())),
                Declarator::Array { .. } => todo!(),
            }
        }
        Declarator::Array { decl, size } => {
            let derived_type = VarType::Array {
                element: Box::new(base_type),
                size,
            };
            process_declarator(decl.map(|b| *b), derived_type)
        }
    }
}

impl<'a> Parser<'a> {
    fn atomic<T, E>(&mut self, f: impl FnOnce(&mut Self) -> Result<T, E>) -> Result<T, E> {
        let index = self.index;
        match f(self) {
            Ok(t) => Ok(t),
            Err(err) => {
                self.index = index;
                Err(err)
            }
        }
    }

    fn many0<T, E>(&mut self, mut f: impl FnMut(&mut Self) -> Result<T, E>) -> Vec<T> {
        let mut res = Vec::new();

        while let Ok(t) = self.atomic(&mut f) {
            res.push(t);
        }

        res
    }

    fn many1<T, E>(&mut self, mut f: impl FnMut(&mut Self) -> Result<T, E>) -> Result<Vec<T>, E> {
        let mut res = Vec::new();
        res.push(self.atomic(&mut f)?);

        while let Ok(t) = self.atomic(&mut f) {
            res.push(t);
        }

        Ok(res)
    }

    fn expect(&mut self, token: Token) -> Result<&Spanned<Token>, Error> {
        if let Some(spanned) = self.tokens.get(self.index) {
            if spanned.data == token {
                self.index += 1;
                Ok(&self.tokens[self.index - 1])
            } else {
                Err(Error::Unexpected(
                    spanned.clone(),
                    ExpectedToken::Token(token),
                ))
            }
        } else {
            Err(Error::UnexpectedEof)
        }
    }

    fn expect_ident(&mut self) -> Result<Spanned<EcoString>, Error> {
        if let Some(spanned) = self.tokens.get(self.index) {
            if let Token::Ident(t) = &spanned.data {
                self.index += 1;
                Ok(Spanned {
                    data: t.clone(),
                    span: spanned.span.clone(),
                })
            } else {
                Err(Error::Unexpected(spanned.clone(), ExpectedToken::Ident))
            }
        } else {
            Err(Error::UnexpectedEof)
        }
    }

    fn expect_constant(&mut self) -> Result<Spanned<Constant>, Error> {
        if let Some(spanned) = self.tokens.get(self.index) {
            if let Token::Constant(c) = &spanned.data {
                self.index += 1;
                Ok(Spanned {
                    data: c.clone(),
                    span: spanned.span.clone(),
                })
            } else {
                Err(Error::Unexpected(spanned.clone(), ExpectedToken::Constant))
            }
        } else {
            Err(Error::UnexpectedEof)
        }
    }

    fn expect_eof(&mut self) -> Result<(), Error> {
        if let Some(spanned) = self.tokens.get(self.index) {
            Err(Error::Unexpected(spanned.clone(), ExpectedToken::Eof))
        } else {
            Ok(())
        }
    }

    fn expect_block(&mut self) -> Result<Block, Error> {
        self.expect(Token::OpenBrace)?;
        let mut body = Vec::new();
        while !matches!(
            self.peek(),
            Some(Spanned {
                data: Token::CloseBrace,
                ..
            })
        ) {
            let index = self.index;

            let err_decl = match self.parse_declaration() {
                Ok(decl) => {
                    body.push(BlockItem::Declaration(decl));
                    continue;
                }
                Err(err) => err,
            };

            let index_decl = self.index;

            self.index = index;

            let err_stmt = match self.parse_statement() {
                Ok(stmt) => {
                    body.push(BlockItem::Statement(stmt));
                    continue;
                }
                Err(err) => err,
            };

            let index_stmt = self.index;

            if index_decl > index_stmt {
                return Err(err_decl);
            } else {
                return Err(err_stmt);
            }
        }
        self.expect(Token::CloseBrace)?;

        Ok(Block(body))
    }

    fn expect_for_init(&mut self) -> Result<Option<ForInit>, Error> {
        if self.expect(Token::SemiColon).is_ok() {
            return Ok(None);
        }
        if let Ok(decl) = self.atomic(|s| s.parse_var_decl()) {
            Ok(Some(ForInit::VarDecl(decl)))
        } else {
            let exp = self.parse_expression(0)?;
            self.expect(Token::SemiColon)?;
            Ok(Some(ForInit::Expression(exp)))
        }
    }

    fn peek(&self) -> Option<&Spanned<Token>> {
        self.tokens.get(self.index)
    }

    fn advance(&mut self) {
        self.index += 1;
        debug_assert!(self.index <= self.tokens.len());
    }

    fn parse_program(&mut self) -> Result<Program, Error> {
        let mut decls = Vec::new();
        loop {
            if self.expect_eof().is_ok() {
                break;
            }
            decls.push(self.parse_declaration()?);
        }
        Ok(Program { decls })
    }

    fn parse_statement(&mut self) -> Result<Statement, Error> {
        match self.peek() {
            Some(Spanned {
                data: Token::Return,
                ..
            }) => {
                self.advance();
                let expr = self.atomic(|s| s.parse_expression(0));
                self.expect(Token::SemiColon)?;
                Ok(Statement::Return(expr.ok()))
            }
            Some(Spanned {
                data: Token::SemiColon,
                ..
            }) => {
                self.advance();
                Ok(Statement::Null)
            }
            Some(Spanned {
                data: Token::If, ..
            }) => {
                self.advance();
                self.expect(Token::OpenParen)?;
                let condition = self.parse_expression(0)?;
                self.expect(Token::CloseParen)?;
                let then_branch = Box::new(self.parse_statement()?);
                let else_branch = if let Some(Spanned {
                    data: Token::Else, ..
                }) = self.peek()
                {
                    self.advance();
                    Some(Box::new(self.parse_statement()?))
                } else {
                    None
                };
                Ok(Statement::If {
                    condition,
                    then_branch,
                    else_branch,
                })
            }
            Some(Spanned {
                data: Token::OpenBrace,
                ..
            }) => {
                let block = self.expect_block()?;
                Ok(Statement::Compound(block))
            }
            Some(Spanned {
                data: Token::Break,
                span,
            }) => {
                let span = span.clone();
                self.advance();
                self.expect(Token::SemiColon)?;
                Ok(Statement::Break {
                    label: "!!!dummy_break_label!!!".into(),
                    span,
                })
            }
            Some(Spanned {
                data: Token::Continue,
                span,
            }) => {
                let span = span.clone();
                self.advance();
                self.expect(Token::SemiColon)?;
                Ok(Statement::Continue {
                    label: "!!!dummy_continue_label!!!".into(),
                    span,
                })
            }
            Some(Spanned {
                data: Token::While, ..
            }) => {
                self.advance();
                self.expect(Token::OpenParen)?;
                let condition = self.parse_expression(0)?;
                self.expect(Token::CloseParen)?;
                let body = Box::new(self.parse_statement()?);
                Ok(Statement::While {
                    label: "!!!dummy_while_label!!!".into(),
                    condition,
                    body,
                })
            }
            Some(Spanned {
                data: Token::Do, ..
            }) => {
                self.advance();
                let body = Box::new(self.parse_statement()?);
                self.expect(Token::While)?;
                self.expect(Token::OpenParen)?;
                let condition = self.parse_expression(0)?;
                self.expect(Token::CloseParen)?;
                self.expect(Token::SemiColon)?;
                Ok(Statement::DoWhile {
                    label: "!!!dummy_dowhile_label!!!".into(),
                    condition,
                    body,
                })
            }
            Some(Spanned {
                data: Token::For, ..
            }) => {
                self.advance();
                self.expect(Token::OpenParen)?;
                let init = self.expect_for_init()?;
                let condition = if self.expect(Token::SemiColon).is_ok() {
                    None
                } else {
                    let cond = Some(self.parse_expression(0)?);
                    self.expect(Token::SemiColon)?;
                    cond
                };
                let step = if self.expect(Token::CloseParen).is_ok() {
                    None
                } else {
                    let step = Some(self.parse_expression(0)?);
                    self.expect(Token::CloseParen)?;
                    step
                };
                let body = Box::new(self.parse_statement()?);
                Ok(Statement::For {
                    label: "!!!dummy_for_label!!!".into(),
                    init,
                    condition,
                    step,
                    body,
                })
            }
            Some(_) => {
                let exp = self.parse_expression(0)?;
                self.expect(Token::SemiColon)?;
                Ok(Statement::Expression(exp))
            }
            None => Err(Error::UnexpectedEof),
        }
    }

    fn parse_declarator(&mut self) -> Result<Spanned<Declarator>, Error> {
        if let Ok(Spanned { span: aspan, .. }) = self.expect(Token::Asterisk) {
            let aspan = aspan.clone();
            let Spanned { data, span } = self.parse_declarator()?;
            Ok(Spanned {
                data: Declarator::Pointer(Spanned {
                    data: Box::new(data),
                    span: span.clone(),
                }),
                span: aspan.start..span.end,
            })
        } else {
            self.parse_direct_declarator()
        }
    }

    fn parse_direct_declarator(&mut self) -> Result<Spanned<Declarator>, Error> {
        let mut decl = self.parse_simple_declarator()?;

        if let Ok(params) = self.atomic(|s| s.parse_param_list()) {
            let span = decl.span.start..self.tokens[self.index - 1].span.end;
            return Ok(Spanned {
                data: Declarator::Fun {
                    params,
                    decl: decl.map(Box::new),
                },
                span,
            });
        }

        let sizes = self.many0(|s| s.parse_square_constant());

        for s in sizes {
            decl = Spanned {
                data: Declarator::Array {
                    decl: decl.map(Box::new),
                    size: s.data,
                },
                span: s.span,
            };
        }

        Ok(decl)
    }

    fn parse_square_constant(&mut self) -> Result<Spanned<usize>, Error> {
        let start = self.expect(Token::OpenSquareBracket)?.span.start;
        let c = self.expect_constant()?;
        let index = match c.data {
            Constant::Integer { value, .. } => value as usize,
            Constant::Char(c) => c as usize,
            _ => return Err(Error::BadArrayLength(c.span.clone())),
        };
        let end = self.expect(Token::CloseSquareBracket)?.span.end;

        Ok(Spanned {
            data: index,
            span: start..end,
        })
    }

    fn parse_param_list(&mut self) -> Result<Vec<ParamInfo>, Error> {
        let mut params = Vec::new();
        self.expect(Token::OpenParen)?;

        let is_void = self
            .atomic(|s| {
                s.expect(Token::Void)?;
                s.expect(Token::CloseParen)?;
                Ok::<_, Error>(())
            })
            .is_ok();

        if is_void {
            return Ok(params);
        }

        loop {
            params.push(self.parse_param()?);
            if self.expect(Token::Comma).is_err() {
                self.expect(Token::CloseParen)?;
                break;
            }
        }
        Ok(params)
    }

    fn parse_param(&mut self) -> Result<ParamInfo, Error> {
        let ty = self.parse_specifiers(false)?.0;
        let decl = self.parse_declarator()?;
        Ok(ParamInfo { ty, decl })
    }

    fn parse_simple_declarator(&mut self) -> Result<Spanned<Declarator>, Error> {
        match self.peek() {
            Some(Spanned {
                data: Token::Ident(ident),
                span,
            }) => {
                let ident = ident.clone();
                let decl = Declarator::Ident(ident);
                let span = span.clone();
                self.advance();
                Ok(Spanned {
                    data: decl,
                    span: span.clone(),
                })
            }
            Some(Spanned {
                data: Token::OpenParen,
                span,
            }) => {
                let start = span.start;
                self.advance();
                let decl = self.parse_declarator()?;
                let end = self.expect(Token::CloseParen)?.span.end;
                Ok(Spanned {
                    data: decl.data,
                    span: start..end,
                })
            }
            Some(s) => Err(Error::Unexpected(s.clone(), ExpectedToken::Declarator)),
            _ => Err(Error::UnexpectedEof),
        }
    }

    fn parse_specifiers(
        &mut self,
        allow_storage_class: bool,
    ) -> Result<(VarType, Option<StorageClass>), Error> {
        let mut ty = Vec::new();
        let mut storage_class = None;
        let start = if let Some(spanned) = self.peek() {
            spanned.span.start
        } else {
            return Err(Error::UnexpectedEof);
        };

        let mut end = start;

        while let Some(s) = self.peek() {
            match &s.data {
                Token::Void => {
                    ty.push(s.clone().map(|_| TypeSpecifier::Void));
                    end = s.span.end;
                    self.advance();
                }
                Token::Char => {
                    ty.push(s.clone().map(|_| TypeSpecifier::Char));
                    end = s.span.end;
                    self.advance();
                }
                Token::Int => {
                    ty.push(s.clone().map(|_| TypeSpecifier::Int));
                    end = s.span.end;
                    self.advance();
                }
                Token::Long => {
                    ty.push(s.clone().map(|_| TypeSpecifier::Long));
                    end = s.span.end;
                    self.advance();
                }
                Token::Signed => {
                    ty.push(s.clone().map(|_| TypeSpecifier::Signed));
                    end = s.span.end;
                    self.advance();
                }
                Token::Unsigned => {
                    ty.push(s.clone().map(|_| TypeSpecifier::Unsigned));
                    end = s.span.end;
                    self.advance();
                }
                Token::Double => {
                    ty.push(s.clone().map(|_| TypeSpecifier::Double));
                    end = s.span.end;
                    self.advance();
                }
                Token::Struct => {
                    self.advance();
                    let tag = self.expect_ident()?;
                    end = tag.span.end;
                    ty.push(Spanned {
                        data: TypeSpecifier::Struct(tag.data.clone()),
                        span: tag.span,
                    });
                }
                Token::Static => {
                    if storage_class.is_some() || !allow_storage_class {
                        return Err(Error::ConflictingSpecifier(s.span.clone()));
                    }
                    end = s.span.end;
                    storage_class = Some(StorageClass::Static);
                    self.advance();
                }
                Token::Extern => {
                    if storage_class.is_some() || !allow_storage_class {
                        return Err(Error::ConflictingSpecifier(s.span.clone()));
                    }
                    end = s.span.end;
                    storage_class = Some(StorageClass::Extern);
                    self.advance();
                }
                _ => break,
            }
        }

        match ty.len() {
            0 => Err(Error::NoTypeSpecifier(start..end)),
            _ => Ok((solve_type_specifier(&ty)?, storage_class)),
        }
    }

    fn parse_initializer(&mut self) -> Result<Initializer, Error> {
        if self.expect(Token::OpenBrace).is_ok() {
            let mut inits = Vec::new();

            loop {
                if self.expect(Token::CloseBrace).is_ok() {
                    break;
                }

                inits.push(self.parse_initializer()?);

                if self.expect(Token::Comma).is_err() {
                    self.expect(Token::CloseBrace)?;
                    break;
                }
            }

            if inits.is_empty() {
                todo!()
            }

            Ok(Initializer::CompoundInit(inits))
        } else {
            let exp = self.parse_expression(0)?;
            Ok(Initializer::SingleInit(exp))
        }
    }

    fn parse_var_decl(&mut self) -> Result<VarDecl, Error> {
        let (ty, storage_class) = self.parse_specifiers(true)?;
        let decl = self.parse_declarator()?;
        let (ident, ty, _) = process_declarator(decl, ty)?;

        let ty = match ty {
            Ty::Var(ty) => ty,
            Ty::Fun(_) => return Err(Error::NotVarType(ident.span.clone())),
        };

        if self.expect(Token::Equal).is_ok() {
            let init = self.parse_initializer()?;
            self.expect(Token::SemiColon)?;
            Ok(VarDecl {
                ident,
                ty,
                init: Some(init),
                storage_class,
            })
        } else {
            self.expect(Token::SemiColon)?;
            Ok(VarDecl {
                ident,
                ty,
                init: None,
                storage_class,
            })
        }
    }

    fn parse_fun_decl(&mut self) -> Result<FunDecl, Error> {
        let (return_type, storage_class) = self.parse_specifiers(true)?;
        let decl = self.parse_declarator()?;
        let span = decl.span.clone();

        let (name, ty, params) = process_declarator(decl, return_type)?;

        let ty = match ty {
            Ty::Var(_) => return Err(Error::NotFunType(span.clone())),
            Ty::Fun(ft) => ft,
        };

        let body = if self.expect(Token::SemiColon).is_ok() {
            None
        } else {
            Some(self.expect_block()?)
        };

        Ok(FunDecl {
            name,
            params,
            ty,
            body,
            storage_class,
        })
    }

    fn parse_declaration(&mut self) -> Result<Declaration, Error> {
        let index = self.index;
        match self.parse_var_decl() {
            Ok(decl) => Ok(Declaration::VarDecl(decl)),
            Err(var_err) => {
                let var_decl_fail = self.index;
                self.index = index;
                match self.parse_fun_decl() {
                    Ok(decl) => Ok(Declaration::FunDecl(decl)),
                    Err(fun_err) => {
                        let fun_decl_fail = self.index;
                        self.index = index;
                        match self.parse_struct_decl() {
                            Ok(decl) => Ok(Declaration::StructDecl(decl)),
                            Err(struct_err) => {
                                let struct_decl_fail = self.index;
                                self.index = index;
                                if var_decl_fail > fun_decl_fail && var_decl_fail > struct_decl_fail
                                {
                                    Err(var_err)
                                } else if fun_decl_fail > var_decl_fail
                                    && fun_decl_fail > struct_decl_fail
                                {
                                    Err(fun_err)
                                } else {
                                    Err(struct_err)
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    fn parse_primary_exp(&mut self) -> Result<Expression, Error> {
        if let Some(token) = self.peek() {
            match &token.data {
                Token::Constant(constant) => match constant {
                    Constant::Integer { value, suffix } => match suffix {
                        Suffix { u: true, l: true } => {
                            let constant = token.clone().map(|_| Const::Ulong(*value));
                            self.advance();
                            Ok(Expression::Constant(constant))
                        }
                        Suffix { u: true, l: false } => {
                            if let Ok(value) = u32::try_from(*value) {
                                let constant = token.clone().map(|_| Const::Uint(value));
                                self.advance();
                                Ok(Expression::Constant(constant))
                            } else {
                                let constant = token.clone().map(|_| Const::Ulong(*value));
                                self.advance();
                                Ok(Expression::Constant(constant))
                            }
                        }
                        Suffix { u: false, l: true } => {
                            let constant = token.clone().map(|_| Const::Long(*value as i64));
                            self.advance();
                            Ok(Expression::Constant(constant))
                        }
                        Suffix { u: false, l: false } => {
                            if let Ok(value) = i32::try_from(*value) {
                                let constant = token.clone().map(|_| Const::Int(value));
                                self.advance();
                                Ok(Expression::Constant(constant))
                            } else {
                                let constant = token.clone().map(|_| Const::Long(*value as i64));
                                self.advance();
                                Ok(Expression::Constant(constant))
                            }
                        }
                    },
                    Constant::Float(value) => {
                        let constant = token.clone().map(|_| Const::Double(*value));
                        self.advance();
                        Ok(Expression::Constant(constant))
                    }
                    Constant::Char(value) => {
                        let constant = token.clone().map(|_| Const::Int(*value as _));
                        self.advance();
                        Ok(Expression::Constant(constant))
                    }
                    Constant::String(value) => {
                        let start = token.span.start;
                        let mut end = token.span.end;
                        let mut s = value.clone();
                        self.advance();
                        // adjacent string literals must be concatenated
                        while let Some(Spanned {
                            data: Token::Constant(Constant::String(s2)),
                            span,
                        }) = self.peek()
                        {
                            s.extend(s2);
                            end = span.end;
                            self.advance();
                        }

                        let len = s.len();

                        Ok(Expression::String(
                            Spanned {
                                data: s,
                                span: start..end,
                            },
                            VarType::Array {
                                element: Box::new(BaseType::Char.into()),
                                size: len + 1,
                            },
                        ))
                    }
                },
                Token::OpenParen => {
                    self.advance();
                    let exp = self.parse_expression(0)?;
                    self.expect(Token::CloseParen)?;
                    Ok(exp)
                }
                Token::Ident(ident) => {
                    let ident = ident.clone();
                    let span = token.span.clone();

                    self.advance();

                    if let Some(Spanned {
                        data: Token::OpenParen,
                        ..
                    }) = self.peek()
                    {
                        self.advance();
                        let mut args = Vec::new();
                        if self.expect(Token::CloseParen).is_err() {
                            loop {
                                args.push(self.parse_expression(0)?);
                                if self.expect(Token::Comma).is_err() {
                                    break;
                                }
                            }
                            self.expect(Token::CloseParen)?;
                        }

                        Ok(Expression::FunctionCall {
                            name: Spanned { data: ident, span },
                            args,
                            ty: VarType::Void,
                        })
                    } else {
                        Ok(Expression::Var(
                            Spanned {
                                data: ident,
                                span: span.clone(),
                            },
                            VarType::Void,
                        ))
                    }
                }
                _ => Err(Error::MalformedExpression(token.clone())),
            }
        } else {
            Err(Error::UnexpectedEof)
        }
    }

    fn parse_postfix_exp(&mut self) -> Result<Expression, Error> {
        let mut exp = self.parse_primary_exp()?;
        let postfixes = self.many0(|s| s.parse_postfix_op());

        for op in postfixes {
            match op {
                PostfixOp::Subscript(index) => {
                    exp = Expression::Subscript {
                        array: Box::new(exp),
                        index: Box::new(index),
                        ty: VarType::Void,
                    };
                }
                PostfixOp::Dot(ident) => {
                    exp = Expression::Dot {
                        structure: Box::new(exp),
                        member: ident,
                        ty: VarType::Void,
                    };
                }
                PostfixOp::Arrow(ident) => {
                    exp = Expression::Arrow {
                        pointer: Box::new(exp),
                        member: ident,
                        ty: VarType::Void,
                    }
                }
            }
        }

        Ok(exp)
    }

    fn parse_postfix_op(&mut self) -> Result<PostfixOp, ()> {
        match self.peek() {
            Some(Spanned {
                data: Token::OpenSquareBracket,
                ..
            }) => {
                self.advance();
                let exp = self.parse_expression(0)?;
                self.expect(Token::CloseSquareBracket)?;
                Ok(PostfixOp::Subscript(exp))
            }
            Some(Spanned {
                data: Token::Dot, ..
            }) => {
                self.advance();
                let ident = self.expect_ident()?;
                Ok(PostfixOp::Dot(ident))
            }
            Some(Spanned {
                data: Token::Arrow, ..
            }) => {
                self.advance();
                let ident = self.expect_ident()?;
                Ok(PostfixOp::Arrow(ident))
            }
            _ => Err(()),
        }
    }

    fn parse_cast_exp(&mut self) -> Result<Expression, Error> {
        self.atomic(|s| {
            s.expect(Token::OpenParen)?;
            let ty = s.parse_type_name()?;
            s.expect(Token::CloseParen)?;
            let exp = s.parse_cast_exp()?;
            Ok::<_, Error>(Expression::Cast {
                target: ty,
                exp: Box::new(exp),
            })
        })
        .or_else(|_| self.parse_unary_exp())
    }

    fn parse_unary_exp(&mut self) -> Result<Expression, Error> {
        if let Some(token) = self.peek() {
            match &token.data {
                _ if UnaryOp::try_from(&token.data).is_ok() => {
                    let op = UnaryOp::try_from(&token.data).unwrap();
                    let op = token.clone().map(|_| op);
                    self.advance();
                    let exp = self.parse_cast_exp()?;
                    Ok(Expression::Unary {
                        op,
                        exp: Box::new(exp),
                        // Fixed in type check pass
                        ty: VarType::Void,
                    })
                }
                Token::Ampersands => {
                    self.advance();
                    let exp = self.parse_cast_exp()?;
                    Ok(Expression::AddrOf {
                        exp: Box::new(exp),
                        // Fixed in type check pass
                        ty: VarType::Void,
                    })
                }
                Token::Asterisk => {
                    self.advance();
                    let exp = self.parse_cast_exp()?;
                    Ok(Expression::Dereference(Box::new(exp)))
                }
                Token::Sizeof => {
                    self.advance();
                    if let Ok(exp) = self.atomic(|s| s.parse_unary_exp()) {
                        Ok(Expression::Sizeof(Box::new(exp)))
                    } else {
                        let start = self.expect(Token::OpenParen)?.span.start;
                        let ty = self.parse_type_name()?;
                        let end = self.expect(Token::CloseParen)?.span.end;
                        Ok(Expression::SizeofType(Spanned {
                            data: ty,
                            span: start..end,
                        }))
                    }
                }
                _ => self.parse_postfix_exp(),
            }
        } else {
            Err(Error::UnexpectedEof)
        }
    }

    fn parse_abstract_declarator(&mut self) -> Result<Spanned<Declarator>, Error> {
        if let Ok(Spanned { span: aspan, .. }) = self.expect(Token::Asterisk) {
            let aspan = aspan.clone();
            if let Ok(Spanned { data, span }) = self.atomic(|s| s.parse_abstract_declarator()) {
                Ok(Spanned {
                    data: Declarator::Pointer(Spanned {
                        data: Box::new(data),
                        span: span.clone(),
                    }),
                    span: aspan.start..span.end,
                })
            } else {
                Ok(Spanned {
                    data: Declarator::Pointer(Spanned {
                        data: Box::new(Declarator::Ident("".into())),
                        span: aspan.clone(),
                    }),
                    span: aspan.clone(),
                })
            }
        } else {
            self.parse_direct_abstract_declarator()
        }
    }

    fn parse_direct_abstract_declarator(&mut self) -> Result<Spanned<Declarator>, Error> {
        let r = self.atomic(|s| {
            s.expect(Token::OpenParen)?;
            let mut decl = s.parse_abstract_declarator()?;
            s.expect(Token::CloseParen)?;
            let sizes = s.many0(|s| s.parse_square_constant());

            for s in sizes {
                decl = Spanned {
                    data: Declarator::Array {
                        decl: decl.map(Box::new),
                        size: s.data,
                    },
                    span: s.span,
                };
            }
            Ok::<_, Error>(decl)
        });

        if let Ok(r) = r {
            Ok(r)
        } else {
            let sizes = self.many1(|s| s.parse_square_constant())?;
            let mut decl = Spanned {
                data: Declarator::Ident("".into()),
                span: 0..0,
            };
            for s in sizes {
                decl = Spanned {
                    data: Declarator::Array {
                        decl: decl.map(Box::new),
                        size: s.data,
                    },
                    span: s.span,
                };
            }

            Ok(decl)
        }
    }

    fn parse_type_name(&mut self) -> Result<VarType, Error> {
        let base_type = self.parse_specifiers(false)?.0;
        if let Ok(decl) = self.atomic(|s| s.parse_abstract_declarator()) {
            let span = decl.span.clone();

            let (_, ty, _) = process_declarator(decl, base_type)?;

            match ty {
                Ty::Var(ty) => Ok(ty),
                Ty::Fun(_) => Err(Error::NotVarType(span)),
            }
        } else {
            Ok(base_type)
        }
    }

    fn parse_expression(&mut self, min_prec: usize) -> Result<Expression, Error> {
        let mut left = self.parse_cast_exp()?;
        loop {
            let Some(token) = self.peek() else {
                break;
            };

            enum Op {
                Binary(BinaryOp),
                Assign,
                Condition,
            }

            impl Op {
                fn precedence(&self) -> usize {
                    match self {
                        Self::Binary(op) => op.precedence(),
                        Self::Assign => 1,
                        Self::Condition => 3,
                    }
                }
            }

            let op = match token.data {
                Token::Equal => Op::Assign,
                Token::Question => Op::Condition,
                _ if BinaryOp::try_from(&token.data).is_ok() => {
                    Op::Binary(BinaryOp::try_from(&token.data).unwrap())
                }
                _ => break,
            };

            if op.precedence() >= min_prec {
                self.advance();
                match op {
                    Op::Assign => {
                        let right = self.parse_expression(op.precedence())?;
                        left = Expression::Assignment {
                            lhs: Box::new(left),
                            rhs: Box::new(right),
                        };
                    }
                    Op::Condition => {
                        let then_branch = self.parse_expression(0)?;
                        self.expect(Token::Colon)?;
                        let else_branch = self.parse_expression(op.precedence())?;
                        left = Expression::Conditional {
                            condition: Box::new(left),
                            then_branch: Box::new(then_branch),
                            else_branch: Box::new(else_branch),
                        };
                    }
                    Op::Binary(bin_op) => {
                        let right = self.parse_expression(bin_op.precedence() + 1)?;
                        left = Expression::Binary {
                            op: bin_op,
                            lhs: Box::new(left),
                            rhs: Box::new(right),
                            ty: BaseType::Int.into(),
                        };
                    }
                }
            } else {
                break;
            }
        }
        Ok(left)
    }

    fn parse_struct_decl(&mut self) -> Result<StructDecl, Error> {
        self.expect(Token::Struct)?;
        let tag = self.expect_ident()?;

        let member_decls = self.atomic(|s| {
            s.expect(Token::OpenBrace)?;
            let member_decls = s.many1(|s| s.parse_struct_member())?;
            s.expect(Token::CloseBrace)?;
            Ok::<_, Error>(member_decls)
        });
        self.expect(Token::SemiColon)?;

        Ok(StructDecl {
            tag,
            member_decls: member_decls.unwrap_or_default(),
        })
    }

    fn parse_struct_member(&mut self) -> Result<MemberDecl, Error> {
        let ty = self.parse_specifiers(false)?.0;
        let decl = self.parse_declarator()?;
        let (ident, ty, _) = process_declarator(decl, ty)?;
        self.expect(Token::SemiColon)?;

        let ty = match ty {
            Ty::Fun(_) => return Err(Error::NotVarType(ident.span.clone())),
            Ty::Var(ty) => ty,
        };

        Ok(MemberDecl {
            name: ident.data,
            ty,
        })
    }
}

enum PostfixOp {
    Subscript(Expression),
    Dot(Spanned<EcoString>),
    Arrow(Spanned<EcoString>),
}
