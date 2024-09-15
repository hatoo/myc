use std::hash::Hash;

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
    VarDecl(VarDecl),
    FunDecl(FunDecl),
    StructDecl(StructDecl),
}

#[derive(Debug)]
pub struct VarDecl {
    pub ident: TokenSpanned<EcoString>,
    pub init: Option<Initializer>,
    pub ty: VarType,
    pub storage_class: Option<StorageClass>,
}

#[derive(Debug)]
pub struct FunDecl {
    pub name: TokenSpanned<EcoString>,
    pub params: Vec<TokenSpanned<EcoString>>,
    pub body: Option<Block>,
    pub ty: FunType,
    pub storage_class: Option<StorageClass>,
}

#[derive(Debug)]
pub struct StructDecl {
    pub tag: TokenSpanned<EcoString>,
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
    Goto(TokenSpanned<EcoString>),
    Label {
        label: TokenSpanned<EcoString>,
        statement: Box<Statement>,
    },
    Switch {
        condition: Expression,
        cases: Vec<SwitchCase>,
    },
}

#[derive(Debug)]
pub enum SwitchCaseTag {
    Case(Expression),
    Default,
}

#[derive(Debug)]
pub struct SwitchCase {
    case: SwitchCaseTag,
    body: Block,
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
        }
    }
}

#[derive(Debug, Clone)]
pub enum Expression {
    Var(TokenSpanned<EcoString>, VarType),
    Cast {
        target: VarType,
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
    SizeofType(TokenSpanned<VarType>),
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
            Self::Cast { target, .. } => target,
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
    BitAnd,
    BitOr,
    Xor,
    ShiftLeft,
    ShiftRight,
}

impl BinaryOp {
    fn precedence(&self) -> usize {
        match self {
            Self::Or => 5,
            Self::And => 10,
            Self::BitOr => 11,
            Self::Xor => 12,
            Self::BitAnd => 13,
            Self::Equal | Self::NotEqual => 30,
            Self::LessThan | Self::LessOrEqual | Self::GreaterThan | Self::GreaterOrEqual => 35,
            Self::ShiftLeft | Self::ShiftRight => 40,
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
            Token::Ampersand => Ok(Self::BitAnd),
            Token::Pipe => Ok(Self::BitOr),
            Token::Caret => Ok(Self::Xor),
            Token::TwoLessThan => Ok(Self::ShiftLeft),
            Token::TwoGreaterThan => Ok(Self::ShiftRight),
            _ => Err(()),
        }
    }
}

pub fn parse(tokens: &[span::Spanned<Token>]) -> Result<Program, Error> {
    let mut parser = Parser { tokens, index: 0 };
    parser.parse_program()
}

struct Parser<'a> {
    tokens: &'a [span::Spanned<Token>],
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
        }
    }
}

fn solve_type_specifier(ty: &[TokenSpanned<TypeSpecifier>]) -> Result<VarType, Error> {
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
        return Ok(BaseType::Double.into());
    }

    if matches!(
        ty,
        [TokenSpanned {
            data: TypeSpecifier::Void,
            ..
        }]
    ) {
        return Ok(VarType::Void);
    }

    if let [TokenSpanned {
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

    Ok(base_ty.into())
}

#[derive(Debug)]
enum Declarator {
    Ident(EcoString),
    Pointer(TokenSpanned<Box<Declarator>>),
    Array {
        decl: TokenSpanned<Box<Declarator>>,
        size: usize,
    },
    Fun {
        params: Vec<ParamInfo>,
        decl: TokenSpanned<Box<Declarator>>,
    },
}

#[derive(Debug)]
struct ParamInfo {
    ty: VarType,
    decl: TokenSpanned<Declarator>,
}

#[allow(clippy::type_complexity)]
fn process_declarator(
    decl: TokenSpanned<Declarator>,
    base_type: VarType,
) -> Result<(TokenSpanned<EcoString>, Ty, Vec<TokenSpanned<EcoString>>), Error> {
    let span = decl.span.clone();
    match decl.data {
        Declarator::Ident(name) => Ok((
            TokenSpanned { data: name, span },
            Ty::Var(base_type),
            Vec::new(),
        )),
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

                    Ok((TokenSpanned { data: name, span }, derived_type, param_names))
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

    fn parse_block_item(&mut self) -> Result<BlockItem, Error> {
        let index = self.index;

        let err_decl = match self.parse_declaration() {
            Ok(decl) => {
                return Ok(BlockItem::Declaration(decl));
            }
            Err(err) => err,
        };

        let index_decl = self.index;

        self.index = index;

        let err_stmt = match self.parse_statement() {
            Ok(stmt) => {
                return Ok(BlockItem::Statement(stmt));
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

    fn expect_block(&mut self) -> Result<Block, Error> {
        self.expect(Token::OpenBrace)?;
        let mut body = Vec::new();
        while !matches!(
            self.peek(),
            Ok(TokenSpanned {
                data: Token::CloseBrace,
                ..
            })
        ) {
            body.push(self.parse_block_item()?);
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
        let mut decls = Vec::new();
        loop {
            if self.expect_eof().is_ok() {
                break;
            }
            decls.push(self.parse_declaration()?);
        }
        Ok(Program { decls })
    }

    fn parse_switch_case(&mut self) -> Result<SwitchCase, Error> {
        let case = if self.expect(Token::Case).is_ok() {
            let exp = self.parse_expression(0)?;
            self.expect(Token::Colon)?;
            SwitchCaseTag::Case(exp)
        } else if self.expect(Token::Default).is_ok() {
            self.expect(Token::Colon)?;
            SwitchCaseTag::Default
        } else {
            return Err(Error::Unexpected(
                self.peek()?.map(Clone::clone),
                ExpectedToken::Token(Token::Case),
            ));
        };

        let mut body = Vec::new();
        while !matches!(
            self.peek(),
            Ok(TokenSpanned {
                data: Token::CloseBrace | Token::Case | Token::Default,
                ..
            })
        ) {
            body.push(self.parse_block_item()?);
        }

        Ok(SwitchCase {
            case,
            body: Block(body),
        })
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
                let block = self.expect_block()?;
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
            TokenSpanned {
                data: Token::Goto, ..
            } => {
                self.advance();
                let label = self.expect_ident()?;
                self.expect(Token::SemiColon)?;
                Ok(Statement::Goto(label))
            }
            TokenSpanned {
                data: Token::Switch,
                ..
            } => {
                self.advance();
                self.expect(Token::OpenParen)?;
                let exp = self.parse_expression(0)?;
                self.expect(Token::CloseParen)?;
                self.expect(Token::OpenBrace)?;
                let mut cases = Vec::new();
                loop {
                    if let Ok(case) = self.atomic(|s| s.parse_switch_case()) {
                        cases.push(case);
                    } else {
                        break;
                    }
                }
                self.expect(Token::CloseBrace)?;
                Ok(Statement::Switch {
                    condition: exp,
                    cases,
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
        if let Ok(TokenSpanned { span: aspan, .. }) = self.expect(Token::Asterisk) {
            let aspan = aspan.clone();
            let TokenSpanned { data, span } = self.parse_declarator()?;
            Ok(TokenSpanned {
                data: Declarator::Pointer(TokenSpanned {
                    data: Box::new(data),
                    span: span.clone(),
                }),
                span: aspan.start..span.end,
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
                    params,
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

    fn parse_simple_declarator(&mut self) -> Result<TokenSpanned<Declarator>, Error> {
        match self.peek()? {
            TokenSpanned {
                data: Token::Ident(ident),
                span,
            } => {
                let ident = ident.clone();
                let decl = Declarator::Ident(ident);
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
            s => Err(Error::Unexpected(
                s.map(Clone::clone),
                ExpectedToken::Declarator,
            )),
        }
    }

    fn parse_specifiers(
        &mut self,
        allow_storage_class: bool,
    ) -> Result<(VarType, Option<StorageClass>), Error> {
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
                    self.advance();
                    let tag = self.expect_ident()?;
                    end = tag.span.end;
                    ty.push(TokenSpanned {
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
                // To Pass the test
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
                if let Ok(exp) = self.atomic(|s| s.parse_unary_exp()) {
                    Ok(Expression::Sizeof(Box::new(exp)))
                } else {
                    let start = self.expect(Token::OpenParen)?.span.start;
                    let ty = self.parse_type_name()?;
                    let end = self.expect(Token::CloseParen)?.span.end;
                    Ok(Expression::SizeofType(TokenSpanned {
                        data: ty,
                        span: start..end,
                    }))
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

    fn parse_abstract_declarator(&mut self) -> Result<TokenSpanned<Declarator>, Error> {
        if let Ok(TokenSpanned { span: aspan, .. }) = self.expect(Token::Asterisk) {
            let aspan = aspan.clone();
            if let Ok(TokenSpanned { data, span }) = self.atomic(|s| s.parse_abstract_declarator())
            {
                Ok(TokenSpanned {
                    data: Declarator::Pointer(TokenSpanned {
                        data: Box::new(data),
                        span: span.clone(),
                    }),
                    span: aspan.start..span.end,
                })
            } else {
                Ok(TokenSpanned {
                    data: Declarator::Pointer(TokenSpanned {
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

    fn parse_direct_abstract_declarator(&mut self) -> Result<TokenSpanned<Declarator>, Error> {
        let r = self.atomic(|s| {
            s.expect(Token::OpenParen)?;
            let mut decl = s.parse_abstract_declarator()?;
            s.expect(Token::CloseParen)?;
            let sizes = s.many0(|s| s.parse_square_constant());

            for s in sizes {
                decl = TokenSpanned {
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
            let mut decl = TokenSpanned {
                data: Declarator::Ident("".into()),
                span: 0..0,
            };
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
                Token::CaretEqual => Op::BinaryAssign(BinaryOp::Xor),
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
                        left = Expression::Assignment {
                            lhs: Box::new(left.clone()),
                            rhs: Box::new(Expression::Binary {
                                op: bin_op,
                                lhs: Box::new(left),
                                rhs: Box::new(right),
                                ty: VarType::Void,
                            }),
                        }
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
    Dot(TokenSpanned<EcoString>),
    Arrow(TokenSpanned<EcoString>),
    Call(Vec<Expression>),
    Increment,
    Decrement,
}
