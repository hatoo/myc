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
}

#[derive(Debug)]
pub struct VarDecl {
    pub ident: Spanned<EcoString>,
    pub init: Option<Expression>,
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
    Return(Expression),
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

#[derive(Debug, Clone, Copy)]
pub enum Const {
    Int(i32),
    Long(i64),
    Uint(u32),
    Ulong(u64),
    Double(f64),
}

impl Const {
    pub fn get_int(&self) -> i32 {
        match self {
            Self::Int(i) => *i,
            Self::Uint(i) => *i as i32,
            Self::Long(i) => *i as i32,
            Self::Ulong(i) => *i as i32,
            Self::Double(i) => *i as i32,
        }
    }
    pub fn get_uint(&self) -> u32 {
        match self {
            Self::Int(i) => *i as u32,
            Self::Uint(i) => *i,
            Self::Long(i) => *i as u32,
            Self::Ulong(i) => *i as u32,
            Self::Double(i) => *i as u32,
        }
    }
    pub fn get_long(&self) -> i64 {
        match self {
            Self::Int(i) => *i as i64,
            Self::Uint(i) => *i as i64,
            Self::Long(i) => *i,
            Self::Ulong(i) => *i as i64,
            Self::Double(i) => *i as i64,
        }
    }
    pub fn get_ulong(&self) -> u64 {
        match self {
            Self::Int(i) => *i as u64,
            Self::Uint(i) => *i as u64,
            Self::Long(i) => *i as u64,
            Self::Ulong(i) => *i,
            Self::Double(i) => *i as u64,
        }
    }
    pub fn get_double(&self) -> f64 {
        match self {
            Self::Int(i) => *i as f64,
            Self::Uint(i) => *i as f64,
            Self::Long(i) => *i as f64,
            Self::Ulong(i) => *i as f64,
            Self::Double(i) => *i,
        }
    }

    pub fn get_static_init(&self, ty: VarType) -> StaticInit {
        match ty {
            VarType::Int => StaticInit::Int(self.get_int()),
            VarType::Uint => StaticInit::Uint(self.get_uint()),
            VarType::Long => StaticInit::Long(self.get_long()),
            VarType::Ulong => StaticInit::Ulong(self.get_ulong()),
            VarType::Double => StaticInit::Double(self.get_double()),
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
    AddrOf(Box<Expression>),
}

impl Expression {
    pub fn ty(&self) -> VarType {
        match self {
            Self::Var(_, ty) => *ty,
            Self::Cast { target, .. } => *target,
            Self::Constant(Spanned { data, .. }) => match data {
                Const::Int(_) => VarType::Int,
                Const::Long(_) => VarType::Long,
                Const::Uint(_) => VarType::Uint,
                Const::Ulong(_) => VarType::Ulong,
                Const::Double(_) => VarType::Double,
            },
            Self::Unary { ty, .. } => *ty,
            Self::Binary { ty, .. } => *ty,
            Self::Assignment { lhs, .. } => lhs.ty(),
            Self::Conditional { then_branch, .. } => then_branch.ty(),
            Self::FunctionCall { ty, .. } => *ty,
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
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Ty {
    Var(VarType),
    Fun(FunType),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VarType {
    Int,
    Long,
    Uint,
    Ulong,
    Double,
    Pointer(Box<Ty>),
}

impl VarType {
    pub fn size(&self) -> usize {
        match self {
            Self::Int => 4,
            Self::Uint => 4,
            Self::Long => 8,
            Self::Ulong => 8,
            Self::Double => 8,
            Self::Pointer(_) => 8,
        }
    }

    pub fn is_signed(&self) -> bool {
        match self {
            Self::Int => true,
            Self::Uint => false,
            Self::Long => true,
            Self::Ulong => false,
            Self::Double => false,
            Self::Pointer(_) => false,
        }
    }

    pub fn zero(&self) -> StaticInit {
        match self {
            Self::Int => StaticInit::Int(0),
            Self::Uint => StaticInit::Uint(0),
            Self::Long => StaticInit::Long(0),
            Self::Ulong => StaticInit::Ulong(0),
            Self::Double => StaticInit::Double(0.0),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FunType {
    pub params: Vec<Ty>,
    pub ret: Box<Ty>,
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
    Int,
    Long,
    Unsigned,
    Signed,
    Double,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Unexpected token: {0:?}, expected {1:?}")]
    Unexpected(Spanned<Token>, ExpectedToken),
    #[error("Unexpected Eof")]
    UnexpectedEof,
    #[error(transparent)]
    ParseIntError(#[from] std::num::ParseIntError),
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
    #[error("Function Pointer in parameter is not supported")]
    FunPtrInParam(std::ops::Range<usize>),
    #[error("Can't apply additional type derivation to a function type")]
    ComplexFunType(std::ops::Range<usize>),
}

impl MayHasSpan for Error {
    fn may_span(&self) -> Option<std::ops::Range<usize>> {
        match self {
            Error::Unexpected(spanned, _) => Some(spanned.span.clone()),
            Error::UnexpectedEof => None,
            Error::ParseIntError(_) => None,
            Error::MalformedExpression(spanned) => Some(spanned.span.clone()),
            Error::MalformedBody(spanned) => Some(spanned.span.clone()),
            Error::ConflictingSpecifier(span) => Some(span.clone()),
            Error::NoTypeSpecifier(span) => Some(span.clone()),
            Error::BadTypeSpecifier(span) => Some(span.clone()),
            Error::UnexpectedSpecifier(spanned) => Some(spanned.span.clone()),
        }
    }
}

fn solve_type_specifier(ty: &[Spanned<TypeSpecifier>]) -> Result<VarType, Error> {
    debug_assert!(!ty.is_empty());

    let mut int = false;
    let mut long = false;
    let mut signed = false;
    let mut unsigned = false;

    if matches!(
        ty,
        [Spanned {
            data: TypeSpecifier::Double,
            ..
        }]
    ) {
        return Ok(VarType::Double);
    }

    for s in ty {
        match s.data {
            TypeSpecifier::Int => {
                if int {
                    return Err(Error::ConflictingSpecifier(s.span.clone()));
                }
                int = true;
            }
            TypeSpecifier::Long => {
                if long {
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
        }
    }

    if long {
        if unsigned {
            Ok(VarType::Ulong)
        } else {
            Ok(VarType::Long)
        }
    } else if unsigned {
        Ok(VarType::Uint)
    } else {
        Ok(VarType::Int)
    }
}

enum Declarator {
    Ident(Spanned<EcoString>),
    PointerDeclarator(Box<Declarator>),
    FunDeclarator {
        params: Vec<ParamInfo>,
        decl: Box<Declarator>,
    },
}

struct ParamInfo {
    ty: Ty,
    decl: Declarator,
}

fn process_declarator(
    decl: Declarator,
    base_type: Ty,
) -> Result<(Spanned<EcoString>, Ty, Vec<EcoString>), Error> {
    match decl {
        Declarator::Ident(name) => Ok((name, base_type, Vec::new())),
        Declarator::PointerDeclarator(d) => {
            let derived_type = Ty::Var(VarType::Pointer(Box::new(base_type)));
            process_declarator(*d, derived_type)
        }
        Declarator::FunDeclarator { params, decl } => {
            let mut param_names = Vec::new();
            let mut param_types = Vec::new();

            for ParamInfo { ty, decl } in params {
                let (name, ty, _) = process_declarator(decl, ty)?;

                /*
                if matches!(ty, Ty::Fun(_)) {
                    return Err(Error::FunPtrInParam(name.span.clone()));
                }
                */

                param_types.push(ty);
                param_names.push(name.data.clone());
            }
            match *decl {
                Declarator::Ident(name) => {
                    let derived_type = Ty::Fun(FunType {
                        params: param_types,
                        ret: Box::new(base_type),
                    });

                    Ok((name, derived_type, param_names))
                }
                Declarator::PointerDeclarator(decl) => {
                    let (name, ty, _) = process_declarator(*decl, base_type)?;
                    Ok((name, Ty::Var(VarType::Pointer(Box::new(ty))), param_names))
                }
            }
        }
    }
}

impl<'a> Parser<'a> {
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
        let index = self.index;
        if let Ok(decl) = self.parse_var_decl() {
            Ok(Some(ForInit::VarDecl(decl)))
        } else {
            self.index = index;
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
                let expr = self.parse_expression(0)?;
                self.expect(Token::SemiColon)?;
                Ok(Statement::Return(expr))
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

    fn parse_declarator(&mut self) -> Result<Declarator, Error> {
        if self.expect(Token::Asterisk).is_ok() {
            Ok(Declarator::PointerDeclarator(Box::new(
                self.parse_declarator()?,
            )))
        } else {
            self.parse_direct_declarator()
        }
    }

    fn parse_direct_declarator(&mut self) -> Result<Declarator, Error> {
        let decl = self.parse_simple_declarator()?;
        let index = self.index;

        if let Ok(params) = self.parse_param_list() {
            Ok(Declarator::FunDeclarator {
                params,
                decl: Box::new(decl),
            })
        } else {
            self.index = index;
            Ok(decl)
        }
    }

    fn parse_param_list(&mut self) -> Result<Vec<ParamInfo>, Error> {
        let mut params = Vec::new();
        self.expect(Token::OpenParen)?;
        loop {
            if self.expect(Token::Void).is_ok() {
                self.expect(Token::CloseParen)?;
                break;
            }
            params.push(self.parse_param()?);
            if self.expect(Token::Comma).is_err() {
                self.expect(Token::CloseParen)?;
                break;
            }
        }
        Ok(params)
    }

    fn parse_param(&mut self) -> Result<ParamInfo, Error> {
        let (ty, _) = self.parse_specifiers()?;
        let decl = self.parse_declarator()?;
        Ok(ParamInfo {
            ty: Ty::Var(ty),
            decl,
        })
    }

    fn parse_simple_declarator(&mut self) -> Result<Declarator, Error> {
        match self.peek() {
            Some(Spanned {
                data: Token::Ident(ident),
                span,
            }) => {
                let ident = ident.clone();
                let decl = Declarator::Ident(Spanned {
                    data: ident,
                    span: span.clone(),
                });
                self.advance();
                Ok(decl)
            }
            Some(Spanned {
                data: Token::OpenParen,
                ..
            }) => {
                self.advance();
                let decl = self.parse_declarator()?;
                self.expect(Token::CloseParen)?;
                Ok(decl)
            }
            Some(s) => Err(Error::Unexpected(s.clone(), ExpectedToken::Declarator)),
            _ => Err(Error::UnexpectedEof),
        }
    }

    fn parse_specifiers(&mut self) -> Result<(VarType, Option<StorageClass>), Error> {
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
                Token::Static => {
                    if storage_class.is_some() {
                        return Err(Error::ConflictingSpecifier(s.span.clone()));
                    }
                    end = s.span.end;
                    storage_class = Some(StorageClass::Static);
                    self.advance();
                }
                Token::Extern => {
                    if storage_class.is_some() {
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

    fn parse_var_decl(&mut self) -> Result<VarDecl, Error> {
        let (ty, storage_class) = self.parse_specifiers()?;
        let ident = self.expect_ident()?;
        if self.expect(Token::Equal).is_ok() {
            let exp = self.parse_expression(0)?;
            self.expect(Token::SemiColon)?;
            Ok(VarDecl {
                ident,
                ty,
                init: Some(exp),
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

    fn expect_param_type(&mut self) -> Result<VarType, Error> {
        let mut ty = Vec::new();
        loop {
            if let Some(s) = self.peek() {
                match &s.data {
                    Token::Int => {
                        ty.push(s.clone().map(|_| TypeSpecifier::Int));
                        self.advance();
                    }
                    Token::Long => {
                        ty.push(s.clone().map(|_| TypeSpecifier::Long));
                        self.advance();
                    }
                    Token::Signed => {
                        ty.push(s.clone().map(|_| TypeSpecifier::Signed));
                        self.advance();
                    }
                    Token::Unsigned => {
                        ty.push(s.clone().map(|_| TypeSpecifier::Unsigned));
                        self.advance();
                    }
                    Token::Double => {
                        ty.push(s.clone().map(|_| TypeSpecifier::Double));
                        self.advance();
                    }
                    _ => {
                        if ty.is_empty() {
                            return Err(Error::UnexpectedSpecifier(s.clone()));
                        } else {
                            break;
                        }
                    }
                }
            } else if ty.is_empty() {
                return Err(Error::UnexpectedEof);
            } else {
                break;
            }
        }

        solve_type_specifier(&ty)
    }

    fn parse_fun_decl(&mut self) -> Result<FunDecl, Error> {
        let (return_type, storage_class) = self.parse_specifiers()?;
        let name = self.expect_ident()?;
        self.expect(Token::OpenParen)?;
        let params = self.parse_param_list()?;
        self.expect(Token::CloseParen)?;
        let body = if self.expect(Token::SemiColon).is_ok() {
            None
        } else {
            Some(self.expect_block()?)
        };

        let mut param_ident = Vec::new();
        let mut param_type = Vec::new();

        for (ty, ident) in params {
            param_ident.push(ident);
            param_type.push(ty);
        }

        Ok(FunDecl {
            name,
            params: param_ident,
            ty: FunType {
                params: param_type,
                ret: return_type,
            },
            body,
            storage_class,
        })
    }

    fn parse_declaration(&mut self) -> Result<Declaration, Error> {
        let index = self.index;

        if let Ok(decl) = self.parse_var_decl() {
            Ok(Declaration::VarDecl(decl))
        } else {
            self.index = index;
            Ok(Declaration::FunDecl(self.parse_fun_decl()?))
        }
    }

    fn parse_factor(&mut self) -> Result<Expression, Error> {
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
                },
                _ if UnaryOp::try_from(&token.data).is_ok() => {
                    let op = UnaryOp::try_from(&token.data).unwrap();
                    let op = token.clone().map(|_| op);
                    self.advance();
                    let exp = self.parse_factor()?;
                    Ok(Expression::Unary {
                        op,
                        exp: Box::new(exp),
                        // Fixed in type check pass
                        ty: VarType::Int,
                    })
                }
                Token::OpenParen => {
                    self.advance();
                    let index = self.index;
                    if let Ok(ty) = self.expect_param_type() {
                        if self.expect(Token::CloseParen).is_ok() {
                            let exp = self.parse_factor()?;
                            return Ok(Expression::Cast {
                                target: ty,
                                exp: Box::new(exp),
                            });
                        }
                    }
                    self.index = index;
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
                            ty: VarType::Int,
                        })
                    } else {
                        Ok(Expression::Var(
                            Spanned {
                                data: ident,
                                span: span.clone(),
                            },
                            VarType::Int,
                        ))
                    }
                }
                _ => Err(Error::MalformedExpression(token.clone())),
            }
        } else {
            Err(Error::UnexpectedEof)
        }
    }

    fn parse_expression(&mut self, min_prec: usize) -> Result<Expression, Error> {
        let mut left = self.parse_factor()?;
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
                            ty: VarType::Int,
                        };
                    }
                }
            } else {
                break;
            }
        }
        Ok(left)
    }
}
