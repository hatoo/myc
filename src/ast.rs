use std::{
    collections::{BTreeMap, HashMap},
    hash::Hash,
};

use ecow::EcoString;

use crate::{
    lexer::{Constant, HasTokenSpan, MayHasTokenSpan, Suffix, Token, TokenSpanned},
    semantics::type_check::StaticInit,
    span::{self},
};

#[derive(Debug)]
pub struct Program {
    pub decls: Vec<Declaration>,
}

#[derive(Debug)]
pub enum Declaration {
    Var(VarDecl),
    Fun(FunDecl),
}

#[derive(Debug, Clone)]
pub enum TypeDeclaration {
    Struct(StructDecl),
    Union(UnionDecl),
    Fun {
        ret: Option<Box<TypeDeclaration>>,
        params: Vec<Option<TypeDeclaration>>,
    },
}

impl TypeDeclaration {
    fn var_ty(&self) -> Option<VarType> {
        match self {
            Self::Struct(decl) => Some(VarType::Struct(decl.tag.data.clone())),
            Self::Union(decl) => Some(VarType::Union(decl.tag.data.clone())),
            Self::Fun { .. } => None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct VarDecl {
    pub storage_class: Option<StorageClass>,
    pub type_decl: Option<TypeDeclaration>,
    pub ty: VarType,
    pub ident: TokenSpanned<Option<EcoString>>,
    pub init: Option<Initializer>,
}

#[derive(Debug)]
pub struct FunDecl {
    pub type_decl_ret: Option<TypeDeclaration>,
    pub type_decl_params: Vec<Option<TypeDeclaration>>,
    pub ty: FunType,
    pub name: TokenSpanned<EcoString>,
    pub params: Vec<TokenSpanned<EcoString>>,
    pub body: Option<Block>,
    pub storage_class: Option<StorageClass>,
}

#[derive(Debug, Clone)]
pub struct StructDecl {
    pub tag: TokenSpanned<EcoString>,
    pub member_decls: Vec<VarDecl>,
}

#[derive(Debug, Clone)]
pub struct UnionDecl {
    pub tag: TokenSpanned<EcoString>,
    pub member_decls: Vec<VarDecl>,
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
                Self::SingleInit(Expression::Constant(TokenSpanned::new_null(Const::Char(0))))
            }
            BaseType::SChar => {
                Self::SingleInit(Expression::Constant(TokenSpanned::new_null(Const::Char(0))))
            }
            BaseType::UChar => Self::SingleInit(Expression::Constant(TokenSpanned::new_null(
                Const::UChar(0),
            ))),
            BaseType::Int => {
                Self::SingleInit(Expression::Constant(TokenSpanned::new_null(Const::Int(0))))
            }
            BaseType::Uint => {
                Self::SingleInit(Expression::Constant(TokenSpanned::new_null(Const::Uint(0))))
            }
            BaseType::Long => {
                Self::SingleInit(Expression::Constant(TokenSpanned::new_null(Const::Long(0))))
            }
            BaseType::Ulong => Self::SingleInit(Expression::Constant(TokenSpanned::new_null(
                Const::Ulong(0),
            ))),
            BaseType::Double => Self::SingleInit(Expression::Constant(TokenSpanned::new_null(
                Const::Double(0.0),
            ))),
        }
    }
}

impl MayHasTokenSpan for Initializer {
    fn may_token_span(&self) -> Option<std::ops::Range<usize>> {
        match self {
            Self::SingleInit(exp) => Some(exp.token_span()),
            Self::CompoundInit(inits) => {
                let start = inits.first()?.may_token_span()?.start;
                let end = inits.last()?.may_token_span()?.end;
                Some(start..end)
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq, Clone, Copy)]
pub enum StorageClass {
    Static,
    Extern,
    Typedef,
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
    Goto(TokenSpanned<EcoString>),
    Label {
        label: TokenSpanned<EcoString>,
        statement: Box<Statement>,
    },
    Default {
        statement: Box<Statement>,
        label: EcoString,
        span: std::ops::Range<usize>,
    },
    Case {
        exp: Expression,
        statement: Box<Statement>,
        label: EcoString,
        span: std::ops::Range<usize>,
    },
    Switch {
        exp: Expression,
        statement: Box<Statement>,
        label: EcoString,
        labels: SwitchLabels,
    },
}

#[derive(Debug, Default)]
pub struct SwitchLabels {
    pub cases: BTreeMap<u64, EcoString>,
    pub default: Option<EcoString>,
}

#[derive(Debug, Clone, Copy)]
pub enum Const {
    Char(i8),
    UChar(u8),
    Int(i32),
    Long(i64),
    Uint(u32),
    Ulong(u64),
    Double(f64),
}

impl PartialEq for Const {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Char(a), Self::Char(b)) => a == b,
            (Self::UChar(a), Self::UChar(b)) => a == b,
            (Self::Int(a), Self::Int(b)) => a == b,
            (Self::Uint(a), Self::Uint(b)) => a == b,
            (Self::Long(a), Self::Long(b)) => a == b,
            (Self::Ulong(a), Self::Ulong(b)) => a == b,
            (Self::Double(a), Self::Double(b)) => a.to_bits() == b.to_bits(),
            _ => false,
        }
    }
}
impl Eq for Const {}
impl Hash for Const {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let discriminant = core::mem::discriminant(self);

        match self {
            Self::Char(i) => {
                discriminant.hash(state);
                i.hash(state);
            }
            Self::UChar(i) => {
                discriminant.hash(state);
                i.hash(state);
            }
            Self::Int(i) => {
                discriminant.hash(state);
                i.hash(state);
            }
            Self::Uint(i) => {
                discriminant.hash(state);
                i.hash(state);
            }
            Self::Long(i) => {
                discriminant.hash(state);
                i.hash(state);
            }
            Self::Ulong(i) => {
                discriminant.hash(state);
                i.hash(state);
            }
            Self::Double(i) => {
                discriminant.hash(state);
                // totally fine in this purpose
                i.to_bits().hash(state);
            }
        }
    }
}

macro_rules! as_inner {
    ($c:expr, $t:ty) => {
        match $c {
            Self::Char(i) => *i as $t,
            Self::UChar(i) => *i as $t,
            Self::Int(i) => *i as $t,
            Self::Uint(i) => *i as $t,
            Self::Long(i) => *i as $t,
            Self::Ulong(i) => *i as $t,
            Self::Double(i) => *i as $t,
        }
    };
}

impl Const {
    pub fn get_char(&self) -> i8 {
        as_inner!(self, i8)
    }
    pub fn get_uchar(&self) -> u8 {
        as_inner!(self, u8)
    }
    pub fn get_int(&self) -> i32 {
        as_inner!(self, i32)
    }
    pub fn get_uint(&self) -> u32 {
        as_inner!(self, u32)
    }
    pub fn get_long(&self) -> i64 {
        as_inner!(self, i64)
    }
    pub fn get_ulong(&self) -> u64 {
        as_inner!(self, u64)
    }
    pub fn get_double(&self) -> f64 {
        as_inner!(self, f64)
    }

    pub fn is_zero(&self) -> bool {
        match self {
            Self::Char(i) => *i == 0,
            Self::UChar(i) => *i == 0,
            Self::Int(i) => *i == 0,
            Self::Uint(i) => *i == 0,
            Self::Long(i) => *i == 0,
            Self::Ulong(i) => *i == 0,
            Self::Double(i) => *i == 0.0,
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
            VarType::Union(_) => None,
            VarType::Typedef(_) => panic!("Typedef should be removed before tacky generation"),
        }
    }
}

#[derive(Debug, Clone)]
pub enum Expression {
    Var(TokenSpanned<EcoString>, VarType),
    Cast {
        target: Box<VarDecl>,
        exp: Box<Expression>,
    },
    Constant(TokenSpanned<Const>),
    Unary {
        op: TokenSpanned<UnaryOp>,
        exp: Box<Expression>,
        ty: VarType,
    },
    Binary {
        op: BinaryOp,
        lhs: Box<Expression>,
        rhs: Box<Expression>,
        ty: VarType,
        assign: bool,
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
        callee: Box<Expression>,
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
    String(TokenSpanned<Vec<u8>>, VarType),
    Sizeof(Box<Expression>),
    SizeofType(TokenSpanned<Box<VarDecl>>),
    Dot {
        structure: Box<Expression>,
        member: TokenSpanned<EcoString>,
        ty: VarType,
    },
    Arrow {
        pointer: Box<Expression>,
        member: TokenSpanned<EcoString>,
        ty: VarType,
    },
    Increment {
        exp: Box<Expression>,
        postfix: bool,
    },
    Decrement {
        exp: Box<Expression>,
        postfix: bool,
    },
}

impl Expression {
    pub fn ty(&self) -> &VarType {
        match self {
            Self::Var(_, ty) => ty,
            Self::Cast { target, .. } => &target.ty,
            Self::Constant(TokenSpanned { data, .. }) => match data {
                Const::Int(_) => &VarType::Base(BaseType::Int),
                Const::Long(_) => &VarType::Base(BaseType::Long),
                Const::Uint(_) => &VarType::Base(BaseType::Uint),
                Const::Ulong(_) => &VarType::Base(BaseType::Ulong),
                Const::Double(_) => &VarType::Base(BaseType::Double),
                Const::Char(_) => &VarType::Base(BaseType::Char),
                Const::UChar(_) => &VarType::Base(BaseType::UChar),
            },
            Self::Unary { ty, .. } => ty,
            Self::Binary {
                ty, assign, lhs, ..
            } => {
                if *assign {
                    lhs.ty()
                } else {
                    ty
                }
            }
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
            Self::Increment { exp, .. } | Self::Decrement { exp, .. } => exp.ty(),
        }
    }

    pub fn is_null_pointer_constant(&self) -> bool {
        // TODO: const expr
        matches!(
            self,
            Self::Constant(TokenSpanned {
                data: Const::Int(0),
                ..
            }) | Self::Constant(TokenSpanned {
                data: Const::Uint(0),
                ..
            }) | Self::Constant(TokenSpanned {
                data: Const::Long(0),
                ..
            }) | Self::Constant(TokenSpanned {
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

impl HasTokenSpan for Expression {
    fn token_span(&self) -> std::ops::Range<usize> {
        match self {
            Self::Var(ident, ..) => ident.span.clone(),
            Self::Constant(constant) => constant.span.clone(),
            Self::Unary { op, exp, .. } => op.span.start..exp.token_span().end,
            Self::Binary { lhs, rhs, .. } => lhs.token_span().start..rhs.token_span().end,
            Self::Assignment { lhs, rhs, .. } => lhs.token_span().start..rhs.token_span().end,
            Self::Conditional {
                condition,
                else_branch,
                ..
            } => condition.token_span().start..else_branch.token_span().end,
            Self::FunctionCall { callee, .. } => callee.token_span().clone(),
            Self::Cast { exp, .. } => exp.token_span(),
            Self::Dereference(exp) => exp.token_span(),
            Self::AddrOf { exp, .. } => exp.token_span(),
            Self::Subscript { array, index, .. } => {
                array.token_span().start..index.token_span().end
            }
            Self::String(s, _) => s.span.clone(),
            Self::Sizeof(exp) => exp.token_span(),
            Self::SizeofType(ty) => ty.span.clone(),
            Self::Dot {
                structure, member, ..
            } => structure.token_span().start..member.span.end,
            Self::Arrow {
                pointer, member, ..
            } => pointer.token_span().start..member.span.end,
            Self::Increment { exp, .. } | Self::Decrement { exp, .. } => exp.token_span(),
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

#[derive(Debug, Clone, PartialEq)]
pub enum VarType {
    Void,
    Base(BaseType),
    Pointer(Box<Ty>),
    Array { element: Box<VarType>, size: usize },
    Struct(EcoString),
    Union(EcoString),
    // must be removed before tacky generation
    Typedef(TokenSpanned<EcoString>),
}

impl Eq for VarType {}

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
    pub fn is_union(&self) -> bool {
        matches!(self, Self::Union(_))
    }

    pub fn is_function_pointer(&self) -> bool {
        matches!(self, Self::Pointer(ty) if matches!(**ty, Ty::Fun(_)))
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunType {
    pub params: Vec<VarType>,
    pub variable_length_params: bool,
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
    LogicalAnd,
    LogicalOr,
    Equal,
    NotEqual,
    LessThan,
    LessOrEqual,
    GreaterThan,
    GreaterOrEqual,
    BitAnd,
    BitOr,
    BitXor,
    ShiftLeft,
    ShiftRight,
}

impl BinaryOp {
    fn precedence(&self) -> usize {
        match self {
            Self::LogicalOr => 5,
            Self::LogicalAnd => 10,
            Self::BitOr => 11,
            Self::BitXor => 12,
            Self::BitAnd => 13,
            Self::Equal | Self::NotEqual => 30,
            Self::LessThan | Self::LessOrEqual | Self::GreaterThan | Self::GreaterOrEqual => 35,
            Self::ShiftLeft | Self::ShiftRight => 40,
            Self::Add | Self::Subtract => 45,
            Self::Multiply | Self::Divide | Self::Remainder => 50,
        }
    }

    pub fn is_comparison(&self) -> bool {
        matches!(
            self,
            Self::Equal
                | Self::NotEqual
                | Self::LessThan
                | Self::LessOrEqual
                | Self::GreaterThan
                | Self::GreaterOrEqual
        )
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
            Token::TwoAmpersands => Ok(Self::LogicalAnd),
            Token::TwoPipes => Ok(Self::LogicalOr),
            Token::TwoEquals => Ok(Self::Equal),
            Token::ExclamationEquals => Ok(Self::NotEqual),
            Token::LessThan => Ok(Self::LessThan),
            Token::LessThanEquals => Ok(Self::LessOrEqual),
            Token::GreaterThan => Ok(Self::GreaterThan),
            Token::GreaterThanEquals => Ok(Self::GreaterOrEqual),
            Token::Ampersand => Ok(Self::BitAnd),
            Token::Pipe => Ok(Self::BitOr),
            Token::Caret => Ok(Self::BitXor),
            Token::TwoLessThan => Ok(Self::ShiftLeft),
            Token::TwoGreaterThan => Ok(Self::ShiftRight),
            _ => Err(()),
        }
    }
}

pub fn parse(tokens: &[span::Spanned<Token>]) -> Result<Program, Error> {
    let mut parser = Parser {
        tokens,
        index: 0,
        scope: Default::default(),
        counter: 0,
    };
    parser.parse_program()
}

#[derive(Debug, Default)]
struct Scope {
    vars: Vec<HashMap<EcoString, bool>>,
}

impl Scope {
    fn push(&mut self) {
        self.vars.push(HashMap::new());
    }

    fn pop(&mut self) {
        self.vars.pop();
    }

    fn insert(&mut self, name: EcoString, is_typedef: bool) {
        self.vars.last_mut().unwrap().insert(name, is_typedef);
    }

    fn is_typedef(&self, name: &EcoString) -> bool {
        for scope in self.vars.iter().rev() {
            if let Some(is_typedef) = scope.get(name) {
                return *is_typedef;
            }
        }
        false
    }
}

struct Parser<'a> {
    tokens: &'a [span::Spanned<Token>],
    index: usize,
    scope: Scope,
    counter: usize,
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
    Union(EcoString),
    Typedef(TokenSpanned<EcoString>),
    TypeDecl(TypeDeclaration),
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Unexpected token: {0:?}, expected {1:?}")]
    Unexpected(TokenSpanned<Token>, ExpectedToken),
    #[error("Unexpected Eof")]
    UnexpectedEof,
    #[error("Malformed expression: {0:?}")]
    MalformedExpression(TokenSpanned<Token>),
    #[error("Malformed body: {0:?}")]
    MalformedBody(TokenSpanned<Token>),
    #[error("Conflicting specifier: {0:?}")]
    ConflictingSpecifier(std::ops::Range<usize>),
    #[error("No type specifier")]
    NoTypeSpecifier(std::ops::Range<usize>),
    #[error("Bad type specifier")]
    BadTypeSpecifier(std::ops::Range<usize>),
    #[error("Unexpected specifier")]
    UnexpectedSpecifier(TokenSpanned<Token>),
    #[error("Function type isn't allowed here")]
    NotVarType(std::ops::Range<usize>),
    #[error("Variable type isn't allowed here")]
    NotFunType(std::ops::Range<usize>),
    #[error("Array length must be a constant integer")]
    BadArrayLength(std::ops::Range<usize>),
    #[error("Empty Initializer is not allowed")]
    EmptyInitializer(std::ops::Range<usize>),
    #[error("Function type can't be an array element")]
    FunctionCantBeArrayElement(std::ops::Range<usize>),
    #[error("Variable name is missing")]
    NoVariableName(std::ops::Range<usize>),
    #[error("Variable name is not allowed here")]
    VariableNameNotAllowed(std::ops::Range<usize>),
}

impl From<Error> for () {
    fn from(_: Error) {}
}

impl MayHasTokenSpan for Error {
    fn may_token_span(&self) -> Option<std::ops::Range<usize>> {
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
            Error::EmptyInitializer(span) => Some(span.clone()),
            Error::FunctionCantBeArrayElement(span) => Some(span.clone()),
            Error::NoVariableName(span) => Some(span.clone()),
            Error::VariableNameNotAllowed(span) => Some(span.clone()),
        }
    }
}

fn solve_type_specifier(
    ty: &[TokenSpanned<TypeSpecifier>],
) -> Result<(Option<TypeDeclaration>, VarType), Error> {
    debug_assert!(!ty.is_empty());

    let mut int = false;
    let mut long = false;
    let mut char = false;
    let mut signed = false;
    let mut unsigned = false;

    if matches!(
        ty,
        [TokenSpanned {
            data: TypeSpecifier::Double,
            ..
        }]
    ) {
        return Ok((None, BaseType::Double.into()));
    }

    if matches!(
        ty,
        [TokenSpanned {
            data: TypeSpecifier::Void,
            ..
        }]
    ) {
        return Ok((None, VarType::Void));
    }

    if let [TokenSpanned {
        data: TypeSpecifier::Struct(tag),
        ..
    }] = ty
    {
        return Ok((None, VarType::Struct(tag.clone())));
    }

    if let [TokenSpanned {
        data: TypeSpecifier::Union(tag),
        ..
    }] = ty
    {
        return Ok((None, VarType::Union(tag.clone())));
    }

    if let [TokenSpanned {
        data: TypeSpecifier::Typedef(tag),
        ..
    }] = ty
    {
        return Ok((None, VarType::Typedef(tag.clone())));
    }

    if let [TokenSpanned {
        data: TypeSpecifier::TypeDecl(decl),
        span,
    }] = ty
    {
        return Ok((
            Some(decl.clone()),
            decl.var_ty()
                .ok_or_else(|| Error::BadTypeSpecifier(span.clone()))?,
        ));
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
            TypeSpecifier::Struct(_) | TypeSpecifier::Union(_) => {
                return Err(Error::BadTypeSpecifier(s.span.clone()));
            }
            TypeSpecifier::Typedef(_) => {
                return Err(Error::BadTypeSpecifier(s.span.clone()));
            }
            TypeSpecifier::TypeDecl(_) => {
                return Err(Error::BadTypeSpecifier(s.span.clone()));
            }
        }
    }

    let base_ty = match (char, int, long) {
        (true, _, _) => {
            if signed {
                BaseType::SChar
            } else if unsigned {
                BaseType::UChar
            } else {
                BaseType::Char
            }
        }
        (_, _, true) => {
            if signed {
                BaseType::Long
            } else if unsigned {
                BaseType::Ulong
            } else {
                BaseType::Long
            }
        }
        _ => {
            if signed {
                BaseType::Int
            } else if unsigned {
                BaseType::Uint
            } else {
                BaseType::Int
            }
        }
    };

    Ok((None, base_ty.into()))
}

#[derive(Debug)]
enum Declarator {
    Ident(Option<EcoString>),
    Pointer(TokenSpanned<Box<Declarator>>),
    Array {
        decl: TokenSpanned<Box<Declarator>>,
        size: usize,
    },
    Fun {
        params: Vec<ParamInfo>,
        variable_length_params: bool,
        decl: TokenSpanned<Box<Declarator>>,
    },
}

#[derive(Debug)]
struct ParamInfo {
    type_decl: Option<TypeDeclaration>,
    ty: VarType,
    decl: TokenSpanned<Declarator>,
}

#[derive(Debug)]
struct FunctionParamList {
    params: Vec<ParamInfo>,
    variable_length_params: bool,
}

#[allow(clippy::type_complexity)]
fn process_declarator(
    decl: TokenSpanned<Declarator>,
    base_type: VarType,
) -> Result<
    (
        TokenSpanned<Option<EcoString>>,
        Ty,
        Vec<TokenSpanned<Option<EcoString>>>,
        Vec<Option<TypeDeclaration>>,
    ),
    Error,
> {
    let span = decl.span.clone();
    match decl.data {
        Declarator::Ident(name) => Ok((
            TokenSpanned { data: name, span },
            Ty::Var(base_type),
            Vec::new(),
            Vec::new(),
        )),
        Declarator::Pointer(d) => {
            let derived_type = VarType::Pointer(Box::new(Ty::Var(base_type)));
            process_declarator(d.map(|d| *d), derived_type)
        }
        Declarator::Fun {
            params,
            variable_length_params,
            decl,
        } => {
            let mut param_names = Vec::new();
            let mut param_types = Vec::new();
            let mut type_decl_params = Vec::new();

            for ParamInfo {
                type_decl,
                ty,
                decl,
            } in params
            {
                let (name, ty, _, decl_params) = process_declarator(decl, ty)?;

                let var_ty = match ty {
                    Ty::Fun(_) => return Err(Error::NotVarType(name.span.clone())),
                    Ty::Var(var_ty) => var_ty,
                };

                if var_ty.is_function_pointer() {
                    type_decl_params.push(Some(TypeDeclaration::Fun {
                        ret: type_decl.map(Box::new),
                        params: decl_params,
                    }));
                } else {
                    type_decl_params.push(type_decl);
                }
                param_types.push(var_ty);
                param_names.push(name);
            }
            match *decl.data {
                Declarator::Ident(name) => {
                    let derived_type = Ty::Fun(FunType {
                        params: param_types,
                        variable_length_params,
                        ret: base_type,
                    });

                    Ok((
                        TokenSpanned { data: name, span },
                        derived_type,
                        param_names,
                        type_decl_params,
                    ))
                }
                Declarator::Pointer(decl) => {
                    let fun_ptr = VarType::Pointer(Box::new(Ty::Fun(FunType {
                        params: param_types,
                        variable_length_params,
                        ret: base_type,
                    })));

                    let (name, ty, _, _) = process_declarator(decl.map(|d| *d), fun_ptr)?;
                    Ok((name, ty, param_names, type_decl_params))
                }
                Declarator::Fun { .. } => Err(Error::NotVarType(decl.span.clone())),
                Declarator::Array { .. } => {
                    Err(Error::FunctionCantBeArrayElement(decl.span.clone()))
                }
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
    fn anon_symbol(&mut self, prefix: &str) -> EcoString {
        let name = format!("_anon_{}.{}", prefix, self.counter);
        self.counter += 1;
        name.into()
    }

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

    fn expect(&mut self, token: Token) -> Result<TokenSpanned<&Token>, Error> {
        let spanned = self.peek()?;
        if *spanned.data == token {
            self.index += 1;
            Ok(TokenSpanned {
                data: &self.tokens[self.index - 1].data,
                span: self.index - 1..self.index,
            })
        } else {
            Err(Error::Unexpected(
                TokenSpanned {
                    data: spanned.data.clone(),
                    span: self.index..self.index + 1,
                },
                ExpectedToken::Token(token),
            ))
        }
    }

    fn expect_ident(&mut self) -> Result<TokenSpanned<EcoString>, Error> {
        if let Some(spanned) = self.tokens.get(self.index) {
            if let Token::Ident(t) = &spanned.data {
                self.index += 1;
                Ok(TokenSpanned {
                    data: t.clone(),
                    span: self.index - 1..self.index,
                })
            } else {
                Err(Error::Unexpected(
                    TokenSpanned {
                        data: spanned.data.clone(),
                        span: self.index..self.index + 1,
                    },
                    ExpectedToken::Ident,
                ))
            }
        } else {
            Err(Error::UnexpectedEof)
        }
    }

    fn expect_constant(&mut self) -> Result<TokenSpanned<Constant>, Error> {
        if let Some(spanned) = self.tokens.get(self.index) {
            if let Token::Constant(c) = &spanned.data {
                self.index += 1;
                Ok(TokenSpanned {
                    data: c.clone(),
                    span: self.index - 1..self.index,
                })
            } else {
                Err(Error::Unexpected(
                    TokenSpanned {
                        data: spanned.data.clone(),
                        span: self.index..self.index + 1,
                    },
                    ExpectedToken::Constant,
                ))
            }
        } else {
            Err(Error::UnexpectedEof)
        }
    }

    fn expect_eof(&mut self) -> Result<(), Error> {
        if let Some(spanned) = self.tokens.get(self.index) {
            Err(Error::Unexpected(
                TokenSpanned {
                    data: spanned.data.clone(),
                    span: self.index..self.index + 1,
                },
                ExpectedToken::Eof,
            ))
        } else {
            Ok(())
        }
    }

    fn parse_block(&mut self) -> Result<Block, Error> {
        self.scope.push();
        self.expect(Token::OpenBrace)?;
        let mut body = Vec::new();
        while !matches!(
            self.peek(),
            Ok(TokenSpanned {
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

        self.scope.pop();
        Ok(Block(body))
    }

    fn parse_for_init(&mut self) -> Result<Option<ForInit>, Error> {
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

    fn peek(&self) -> Result<TokenSpanned<&Token>, Error> {
        self.tokens
            .get(self.index)
            .map(|t| TokenSpanned {
                data: &t.data,
                span: self.index..self.index + 1,
            })
            .ok_or(Error::UnexpectedEof)
    }

    fn advance(&mut self) {
        self.index += 1;
        debug_assert!(self.index <= self.tokens.len());
    }

    fn parse_program(&mut self) -> Result<Program, Error> {
        self.scope.push();
        let mut decls = Vec::new();
        loop {
            if self.expect_eof().is_ok() {
                break;
            }
            decls.push(self.parse_declaration()?);
        }
        self.scope.pop();
        Ok(Program { decls })
    }

    fn parse_statement(&mut self) -> Result<Statement, Error> {
        match self.peek()? {
            TokenSpanned {
                data: Token::Return,
                ..
            } => {
                self.advance();
                let expr = self.atomic(|s| s.parse_expression(0));
                let res = self.expect(Token::SemiColon);
                if let Err(err) = res {
                    expr?;
                    return Err(err);
                }

                Ok(Statement::Return(expr.ok()))
            }
            TokenSpanned {
                data: Token::SemiColon,
                ..
            } => {
                self.advance();
                Ok(Statement::Null)
            }
            TokenSpanned {
                data: Token::If, ..
            } => {
                self.advance();
                self.expect(Token::OpenParen)?;
                let condition = self.parse_expression(0)?;
                self.expect(Token::CloseParen)?;
                let then_branch = Box::new(self.parse_statement()?);
                let else_branch = if self.expect(Token::Else).is_ok() {
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
            TokenSpanned {
                data: Token::OpenBrace,
                ..
            } => {
                let block = self.parse_block()?;
                Ok(Statement::Compound(block))
            }
            TokenSpanned {
                data: Token::Break,
                span,
            } => {
                let span = span.clone();
                self.advance();
                self.expect(Token::SemiColon)?;
                Ok(Statement::Break {
                    label: "!!!dummy_break_label!!!".into(),
                    span,
                })
            }
            TokenSpanned {
                data: Token::Continue,
                span,
            } => {
                let span = span.clone();
                self.advance();
                self.expect(Token::SemiColon)?;
                Ok(Statement::Continue {
                    label: "!!!dummy_continue_label!!!".into(),
                    span,
                })
            }
            TokenSpanned {
                data: Token::While, ..
            } => {
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
            TokenSpanned {
                data: Token::Do, ..
            } => {
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
            TokenSpanned {
                data: Token::For, ..
            } => {
                self.advance();
                self.expect(Token::OpenParen)?;
                let init = self.parse_for_init()?;
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
            TokenSpanned {
                data: Token::Goto, ..
            } => {
                self.advance();
                let label = self.expect_ident()?;
                self.expect(Token::SemiColon)?;
                Ok(Statement::Goto(label))
            }
            TokenSpanned {
                data: Token::Default,
                span,
            } => {
                self.advance();
                self.expect(Token::Colon)?;
                let statement = Box::new(self.parse_statement()?);
                Ok(Statement::Default {
                    statement,
                    label: "!!!dummy_default_label!!!".into(),
                    span,
                })
            }
            TokenSpanned {
                data: Token::Case,
                span,
            } => {
                self.advance();
                let exp = self.parse_expression(0)?;
                self.expect(Token::Colon)?;
                let statement = Box::new(self.parse_statement()?);
                Ok(Statement::Case {
                    exp,
                    statement,
                    label: "!!!dummy_case_label!!!".into(),
                    span,
                })
            }
            TokenSpanned {
                data: Token::Switch,
                ..
            } => {
                self.advance();
                self.expect(Token::OpenParen)?;
                let exp = self.parse_expression(0)?;
                self.expect(Token::CloseParen)?;
                let statement = Box::new(self.parse_statement()?);
                Ok(Statement::Switch {
                    exp,
                    statement,
                    label: "!!!dummy_switch_label!!!".into(),
                    labels: Default::default(),
                })
            }
            _ => {
                if let Ok(stmt) = self.atomic(|s| {
                    let label = s.expect_ident()?;
                    s.expect(Token::Colon)?;
                    let stmt = s.parse_statement()?;
                    Ok::<_, Error>(Statement::Label {
                        label,
                        statement: Box::new(stmt),
                    })
                }) {
                    Ok(stmt)
                } else {
                    let exp = self.parse_expression(0)?;
                    self.expect(Token::SemiColon)?;
                    Ok(Statement::Expression(exp))
                }
            }
        }
    }

    fn parse_declarator(&mut self) -> Result<TokenSpanned<Declarator>, Error> {
        if let Ok(TokenSpanned {
            span: asterisk_span,
            ..
        }) = self.expect(Token::Asterisk)
        {
            let asterisk_span = asterisk_span.clone();
            let TokenSpanned { data, span } =
                self.parse_declarator().unwrap_or_else(|_| TokenSpanned {
                    data: Declarator::Ident(None),
                    span: asterisk_span.clone(),
                });
            Ok(TokenSpanned {
                data: Declarator::Pointer(TokenSpanned {
                    data: Box::new(data),
                    span: span.clone(),
                }),
                span: asterisk_span.start..span.end,
            })
        } else {
            self.parse_direct_declarator()
        }
    }

    fn parse_direct_declarator(&mut self) -> Result<TokenSpanned<Declarator>, Error> {
        let mut decl = self.parse_simple_declarator()?;

        if let Ok(params) = self.atomic(|s| s.parse_param_list()) {
            let span = decl.span.start..self.index;
            return Ok(TokenSpanned {
                data: Declarator::Fun {
                    params: params.params,
                    variable_length_params: params.variable_length_params,
                    decl: decl.map(Box::new),
                },
                span,
            });
        }

        let sizes = self.many0(|s| s.parse_square_constant());

        for s in sizes {
            decl = TokenSpanned {
                data: Declarator::Array {
                    decl: decl.map(Box::new),
                    size: s.data,
                },
                span: s.span,
            };
        }

        Ok(decl)
    }

    fn parse_square_constant(&mut self) -> Result<TokenSpanned<usize>, Error> {
        let start = self.expect(Token::OpenSquareBracket)?.span.start;
        let c = self.expect_constant()?;
        let index = match c.data {
            Constant::Integer { value, .. } => value as usize,
            Constant::Char(c) => c as usize,
            _ => return Err(Error::BadArrayLength(c.span.clone())),
        };
        let end = self.expect(Token::CloseSquareBracket)?.span.end;

        Ok(TokenSpanned {
            data: index,
            span: start..end,
        })
    }

    fn parse_param_list(&mut self) -> Result<FunctionParamList, Error> {
        let mut params = Vec::new();
        self.expect(Token::OpenParen)?;

        let is_void = self
            .atomic(|s| {
                let _ = s.expect(Token::Void);
                s.expect(Token::CloseParen)?;
                Ok::<_, Error>(())
            })
            .is_ok();

        if is_void {
            return Ok(FunctionParamList {
                params,
                variable_length_params: false,
            });
        }

        loop {
            if self.expect(Token::ThreeDots).is_ok() {
                self.expect(Token::CloseParen)?;
                return Ok(FunctionParamList {
                    params,
                    variable_length_params: true,
                });
            }
            params.push(self.parse_param()?);
            if self.expect(Token::Comma).is_err() {
                self.expect(Token::CloseParen)?;
                break;
            }
        }
        Ok(FunctionParamList {
            params,
            variable_length_params: false,
        })
    }

    fn parse_param(&mut self) -> Result<ParamInfo, Error> {
        // TODO
        let (type_decl, ty, _) = self.parse_specifiers(false)?;
        let decl = self.parse_declarator()?;
        Ok(ParamInfo {
            type_decl,
            ty,
            decl,
        })
    }

    fn parse_simple_declarator(&mut self) -> Result<TokenSpanned<Declarator>, Error> {
        match self.peek()? {
            TokenSpanned {
                data: Token::Ident(ident),
                span,
            } => {
                let ident = ident.clone();
                let decl = Declarator::Ident(Some(ident));
                let span = span.clone();
                self.advance();
                Ok(TokenSpanned {
                    data: decl,
                    span: span.clone(),
                })
            }
            TokenSpanned {
                data: Token::OpenParen,
                span,
            } => {
                let start = span.start;
                self.advance();
                let decl = self.parse_declarator()?;
                let end = self.expect(Token::CloseParen)?.span.end;
                Ok(TokenSpanned {
                    data: decl.data,
                    span: start..end,
                })
            }
            _ => Ok(TokenSpanned {
                data: Declarator::Ident(None),
                span: self.index..self.index + 1,
            }),
        }
    }

    fn parse_specifiers(
        &mut self,
        allow_storage_class: bool,
    ) -> Result<(Option<TypeDeclaration>, VarType, Option<StorageClass>), Error> {
        let mut ty = Vec::new();
        let mut storage_class = None;
        let start = self.peek()?.span.start;

        let mut end = start;

        while let Ok(s) = self.peek() {
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
                    let start = self.index;
                    if let Ok(decl) = self.atomic(|s| s.parse_struct_decl()) {
                        let span = start..self.index;
                        ty.push(TokenSpanned {
                            data: TypeSpecifier::TypeDecl(TypeDeclaration::Struct(decl)),
                            span,
                        });
                    } else {
                        self.advance();
                        let tag = self.expect_ident()?;
                        end = tag.span.end;
                        ty.push(TokenSpanned {
                            data: TypeSpecifier::Struct(tag.data.clone()),
                            span: tag.span,
                        });
                    }
                }
                Token::Union => {
                    let start = self.index;
                    if let Ok(decl) = self.atomic(|s| s.parse_union_decl()) {
                        let span = start..self.index;
                        ty.push(TokenSpanned {
                            data: TypeSpecifier::TypeDecl(TypeDeclaration::Union(decl)),
                            span,
                        });
                    } else {
                        self.advance();
                        let tag = self.expect_ident()?;
                        end = tag.span.end;
                        ty.push(TokenSpanned {
                            data: TypeSpecifier::Union(tag.data.clone()),
                            span: tag.span,
                        });
                    }
                }
                Token::Ident(ident) if self.scope.is_typedef(ident) => {
                    if ty.is_empty() {
                        ty.push(TokenSpanned {
                            data: TypeSpecifier::Typedef(TokenSpanned {
                                data: ident.clone(),
                                span: s.span.clone(),
                            }),
                            span: s.span,
                        });
                        self.advance();
                    } else {
                        break;
                    }
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
                Token::Typedef => {
                    if storage_class.is_some() || !allow_storage_class {
                        return Err(Error::ConflictingSpecifier(s.span.clone()));
                    }
                    end = s.span.end;
                    storage_class = Some(StorageClass::Typedef);
                    self.advance();
                }
                _ => break,
            }
        }

        match ty.len() {
            0 => Err(Error::NoTypeSpecifier(start..end)),
            _ => {
                let (decl, ty) = solve_type_specifier(&ty)?;
                Ok((decl, ty, storage_class))
            }
        }
    }

    fn parse_initializer(&mut self) -> Result<Initializer, Error> {
        let start = self.index;
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
                return Err(Error::EmptyInitializer(start..self.index));
            }

            Ok(Initializer::CompoundInit(inits))
        } else {
            let exp = self.parse_expression(0)?;
            Ok(Initializer::SingleInit(exp))
        }
    }

    fn parse_var_decl_body(&mut self) -> Result<VarDecl, Error> {
        let (type_decl, ty, storage_class) = self.parse_specifiers(true)?;
        let decl = self.parse_declarator()?;
        let (ident, ty, _, type_decl_params) = process_declarator(decl, ty)?;
        let span = ident.span.clone();

        let ty = match ty {
            Ty::Var(ty) => ty,
            Ty::Fun(_) => return Err(Error::NotVarType(span)),
        };

        let type_decl = if let VarType::Pointer(target) = &ty {
            if let Ty::Fun(_) = target.as_ref() {
                Some(TypeDeclaration::Fun {
                    ret: type_decl.map(Box::new),
                    params: type_decl_params,
                })
            } else {
                type_decl
            }
        } else {
            type_decl
        };

        if let Some(ident) = ident.data.as_ref() {
            self.scope
                .insert(ident.clone(), storage_class == Some(StorageClass::Typedef));
        }

        if self.expect(Token::Equal).is_ok() {
            let init = self.parse_initializer()?;
            Ok(VarDecl {
                storage_class,
                type_decl,
                ty,
                ident,
                init: Some(init),
            })
        } else {
            Ok(VarDecl {
                storage_class,
                type_decl,
                ty,
                ident,
                init: None,
            })
        }
    }

    fn parse_var_decl(&mut self) -> Result<VarDecl, Error> {
        let decl = self.parse_var_decl_body()?;
        self.expect(Token::SemiColon)?;
        Ok(decl)
    }

    fn parse_fun_decl(&mut self) -> Result<FunDecl, Error> {
        let (type_decl, return_type, storage_class) = self.parse_specifiers(true)?;
        let decl = self.parse_declarator()?;
        let span = decl.span.clone();

        let (name, ty, params, type_decl_params) = process_declarator(decl, return_type)?;
        let name = if let TokenSpanned {
            data: Some(data),
            span,
        } = name
        {
            TokenSpanned { data, span }
        } else {
            return Err(Error::NoVariableName(name.span));
        };

        let ty = match ty {
            Ty::Var(_) => return Err(Error::NotFunType(span.clone())),
            Ty::Fun(ft) => ft,
        };

        let body = if self.expect(Token::SemiColon).is_ok() {
            None
        } else {
            Some(self.parse_block()?)
        };

        let params = params
            .into_iter()
            .map(|p| {
                if let TokenSpanned {
                    data: Some(data),
                    span,
                } = p
                {
                    Ok(TokenSpanned { data, span })
                } else if body.is_some() {
                    Err(Error::NoVariableName(p.span))
                } else {
                    Ok(TokenSpanned {
                        data: "".into(),
                        span: p.span,
                    })
                }
            })
            .collect::<Result<Vec<_>, _>>()?;

        self.scope.insert(
            name.data.clone(),
            storage_class == Some(StorageClass::Typedef),
        );

        Ok(FunDecl {
            type_decl_ret: type_decl,
            type_decl_params,
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
            Ok(decl) => Ok(Declaration::Var(decl)),
            Err(var_err) => {
                let var_decl_fail = self.index;
                self.index = index;
                match self.parse_fun_decl() {
                    Ok(decl) => Ok(Declaration::Fun(decl)),
                    Err(fun_err) => {
                        let fun_decl_fail = self.index;
                        if var_decl_fail > fun_decl_fail {
                            self.index = var_decl_fail;
                            Err(var_err)
                        } else {
                            self.index = fun_decl_fail;
                            Err(fun_err)
                        }
                    }
                }
            }
        }
    }

    fn parse_primary_exp(&mut self) -> Result<Expression, Error> {
        let token = self.peek()?;
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
                    while let Ok(TokenSpanned {
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
                        TokenSpanned {
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

                if self.expect(Token::OpenParen).is_ok() {
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
                        callee: Box::new(Expression::Var(
                            TokenSpanned { data: ident, span },
                            VarType::Void,
                        )),
                        args,
                        ty: VarType::Void,
                    })
                } else {
                    Ok(Expression::Var(
                        TokenSpanned {
                            data: ident,
                            span: span.clone(),
                        },
                        VarType::Void,
                    ))
                }
            }
            _ => Err(Error::MalformedExpression(token.map(Clone::clone))),
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
                PostfixOp::Call(args) => {
                    exp = Expression::FunctionCall {
                        callee: Box::new(exp),
                        args,
                        ty: VarType::Void,
                    };
                }
                PostfixOp::Increment => {
                    exp = Expression::Increment {
                        exp: Box::new(exp),
                        postfix: true,
                    };
                }
                PostfixOp::Decrement => {
                    exp = Expression::Decrement {
                        exp: Box::new(exp),
                        postfix: true,
                    };
                }
            }
        }

        Ok(exp)
    }

    fn parse_postfix_op(&mut self) -> Result<PostfixOp, ()> {
        match &self.peek()?.data {
            Token::OpenSquareBracket => {
                self.advance();
                let exp = self.parse_expression(0)?;
                self.expect(Token::CloseSquareBracket)?;
                Ok(PostfixOp::Subscript(exp))
            }
            Token::Dot => {
                self.advance();
                let ident = self.expect_ident()?;
                Ok(PostfixOp::Dot(ident))
            }
            Token::Arrow => {
                self.advance();
                let ident = self.expect_ident()?;
                Ok(PostfixOp::Arrow(ident))
            }
            Token::OpenParen => {
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
                Ok(PostfixOp::Call(args))
            }
            Token::TwoPlus => {
                self.advance();
                Ok(PostfixOp::Increment)
            }
            Token::TwoHyphens => {
                self.advance();
                Ok(PostfixOp::Decrement)
            }

            _ => Err(()),
        }
    }

    fn parse_cast_exp(&mut self) -> Result<Expression, Error> {
        self.atomic(|s| {
            s.expect(Token::OpenParen)?;
            let var_decl = s.parse_var_decl_body()?;
            s.expect(Token::CloseParen)?;
            let exp = s.parse_cast_exp()?;
            Ok::<_, Error>(Expression::Cast {
                target: Box::new(var_decl),
                exp: Box::new(exp),
            })
        })
        .or_else(|_| self.parse_unary_exp())
    }

    fn parse_unary_exp(&mut self) -> Result<Expression, Error> {
        let token = self.peek()?;
        match &token.data {
            _ if UnaryOp::try_from(token.data).is_ok() => {
                let op = UnaryOp::try_from(token.data).unwrap();
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
            Token::Ampersand => {
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
                if let Ok(var_decl) = self.atomic(|s| {
                    let start = s.expect(Token::OpenParen)?.span.start;
                    // TODO: check var_decl
                    let var_decl = s.parse_var_decl_body()?;
                    let end = s.expect(Token::CloseParen)?.span.end;
                    Ok::<_, Error>(TokenSpanned {
                        data: var_decl,
                        span: start..end,
                    })
                }) {
                    Ok(Expression::SizeofType(var_decl.map(Box::new)))
                } else {
                    Ok(Expression::Sizeof(Box::new(self.parse_unary_exp()?)))
                }
            }
            Token::TwoPlus => {
                self.advance();
                let exp = self.parse_cast_exp()?;

                Ok(Expression::Increment {
                    exp: Box::new(exp),
                    postfix: false,
                })
            }
            Token::TwoHyphens => {
                self.advance();
                let exp = self.parse_cast_exp()?;

                Ok(Expression::Decrement {
                    exp: Box::new(exp),
                    postfix: false,
                })
            }
            _ => self.parse_postfix_exp(),
        }
    }

    fn parse_expression(&mut self, min_prec: usize) -> Result<Expression, Error> {
        let mut left = self.parse_cast_exp()?;
        loop {
            let Ok(token) = self.peek() else {
                break;
            };

            enum Op {
                Binary(BinaryOp),
                BinaryAssign(BinaryOp),
                Assign,
                Condition,
            }

            impl Op {
                fn precedence(&self) -> usize {
                    match self {
                        Self::Binary(op) => op.precedence(),
                        Self::Assign | Self::BinaryAssign(_) => 1,
                        Self::Condition => 3,
                    }
                }
            }

            let op = match token.data {
                Token::Equal => Op::Assign,
                Token::Question => Op::Condition,
                Token::PlusEqual => Op::BinaryAssign(BinaryOp::Add),
                Token::MinusEqual => Op::BinaryAssign(BinaryOp::Subtract),
                Token::AsteriskEqual => Op::BinaryAssign(BinaryOp::Multiply),
                Token::SlashEqual => Op::BinaryAssign(BinaryOp::Divide),
                Token::PercentEqual => Op::BinaryAssign(BinaryOp::Remainder),
                Token::AmpersandEqual => Op::BinaryAssign(BinaryOp::BitAnd),
                Token::PipeEqual => Op::BinaryAssign(BinaryOp::BitOr),
                Token::CaretEqual => Op::BinaryAssign(BinaryOp::BitXor),
                Token::TwoLessThanEqual => Op::BinaryAssign(BinaryOp::ShiftLeft),
                Token::TwoGreaterThanEqual => Op::BinaryAssign(BinaryOp::ShiftRight),
                _ if BinaryOp::try_from(token.data).is_ok() => {
                    Op::Binary(BinaryOp::try_from(token.data).unwrap())
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
                    Op::BinaryAssign(bin_op) => {
                        let right = self.parse_expression(op.precedence())?;
                        left = Expression::Binary {
                            op: bin_op,
                            lhs: Box::new(left),
                            rhs: Box::new(right),
                            ty: VarType::Void,
                            assign: true,
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
                            assign: false,
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
        let tag = self.expect_ident().unwrap_or_else(|_| TokenSpanned {
            data: self.anon_symbol("struct"),
            span: self.index..self.index + 1,
        });

        self.expect(Token::OpenBrace)?;
        let member_decls = self.many1(|s| s.parse_var_decl())?;
        self.expect(Token::CloseBrace)?;

        Ok(StructDecl { tag, member_decls })
    }

    fn parse_union_decl(&mut self) -> Result<UnionDecl, Error> {
        self.expect(Token::Union)?;
        let tag = self.expect_ident().unwrap_or_else(|_| TokenSpanned {
            data: self.anon_symbol("union"),
            span: self.index..self.index + 1,
        });

        self.expect(Token::OpenBrace)?;
        let member_decls = self.many1(|s| s.parse_var_decl())?;
        self.expect(Token::CloseBrace)?;

        Ok(UnionDecl { tag, member_decls })
    }
}

enum PostfixOp {
    Subscript(Expression),
    Dot(TokenSpanned<EcoString>),
    Arrow(TokenSpanned<EcoString>),
    Call(Vec<Expression>),
    Increment,
    Decrement,
}
