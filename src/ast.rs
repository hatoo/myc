use ecow::EcoString;

use crate::{
    lexer::Token,
    span::{HasSpan, MayHasSpan, Spanned},
};

#[derive(Debug)]
pub struct Program {
    pub decls: Vec<Declaration>,
}

#[derive(Debug, PartialEq, Eq)]
pub enum StorageClass {
    Static,
    Extern,
}

#[derive(Debug)]
pub struct FunDecl {
    pub name: Spanned<EcoString>,
    pub params: Vec<Spanned<EcoString>>,
    pub body: Option<Block>,
    pub storage_class: Option<StorageClass>,
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

#[derive(Debug, Clone)]
pub enum Expression {
    Var(Spanned<EcoString>),
    Constant(Spanned<i32>),
    Unary {
        op: Spanned<UnaryOp>,
        exp: Box<Expression>,
    },
    Binary {
        op: BinaryOp,
        lhs: Box<Expression>,
        rhs: Box<Expression>,
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
    },
}

impl HasSpan for Expression {
    fn span(&self) -> std::ops::Range<usize> {
        match self {
            Self::Var(ident) => ident.span.clone(),
            Self::Constant(constant) => constant.span.clone(),
            Self::Unary { op, exp } => op.span.start..exp.span().end,
            Self::Binary { lhs, rhs, .. } => lhs.span().start..rhs.span().end,
            Self::Assignment { lhs, rhs } => lhs.span().start..rhs.span().end,
            Self::Conditional {
                condition,
                else_branch,
                ..
            } => condition.span().start..else_branch.span().end,
            Self::FunctionCall { name, .. } => name.span.clone(),
        }
    }
}
#[derive(Debug)]
pub struct VarDecl {
    pub ident: Spanned<EcoString>,
    pub exp: Option<Expression>,
    pub storage_class: Option<StorageClass>,
}

#[derive(Debug)]
pub enum Declaration {
    VarDecl(VarDecl),
    FunDecl(FunDecl),
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
}

impl MayHasSpan for Error {
    fn may_span(&self) -> Option<std::ops::Range<usize>> {
        match self {
            Error::Unexpected(spanned, _) => Some(spanned.span.clone()),
            Error::UnexpectedEof => None,
            Error::ParseIntError(_) => None,
            Error::MalformedExpression(spanned) => Some(spanned.span.clone()),
            Error::MalformedBody(spanned) => Some(spanned.span.clone()),
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
            if let Ok(decl) = self.parse_declaration() {
                body.push(BlockItem::Declaration(decl));
            } else {
                self.index = index;
                body.push(BlockItem::Statement(self.parse_statement()?));
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

    fn expect_specifier(&mut self) -> Result<Spanned<Token>, Error> {
        if let Some(spanned) = self.tokens.get(self.index) {
            match &spanned.data {
                Token::Int | Token::Static | Token::Extern => {
                    self.index += 1;
                    Ok(spanned.clone())
                }
                _ => Err(Error::Unexpected(spanned.clone(), ExpectedToken::Specifier)),
            }
        } else {
            Err(Error::UnexpectedEof)
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

    fn parse_specifiers(&mut self) -> Result<Option<StorageClass>, Error> {
        let mut ty = None;
        let mut storage_class = None;

        loop {
            if let Ok(specifier) = self.expect_specifier() {
                match &specifier.data {
                    Token::Int => {
                        // TODO better error handling
                        if ty.is_some() {
                            return Err(Error::Unexpected(
                                specifier.clone(),
                                ExpectedToken::Specifier,
                            ));
                        }
                        ty = Some(specifier);
                    }
                    Token::Static => {
                        if storage_class.is_some() {
                            return Err(Error::Unexpected(
                                specifier.clone(),
                                ExpectedToken::Specifier,
                            ));
                        }
                        storage_class = Some(StorageClass::Static);
                    }
                    Token::Extern => {
                        if storage_class.is_some() {
                            return Err(Error::Unexpected(
                                specifier.clone(),
                                ExpectedToken::Specifier,
                            ));
                        }
                        storage_class = Some(StorageClass::Extern);
                    }
                    _ => unreachable!(),
                }
            } else {
                break;
            }
        }

        let Some(_ty) = ty else {
            // TODO better error handling
            return Err(Error::UnexpectedEof);
        };

        Ok(storage_class)
    }

    fn parse_var_decl(&mut self) -> Result<VarDecl, Error> {
        let storage_class = self.parse_specifiers()?;
        let ident = self.expect_ident()?;
        if self.expect(Token::Equal).is_ok() {
            let exp = self.parse_expression(0)?;
            self.expect(Token::SemiColon)?;
            Ok(VarDecl {
                ident,
                exp: Some(exp),
                storage_class,
            })
        } else {
            self.expect(Token::SemiColon)?;
            Ok(VarDecl {
                ident,
                exp: None,
                storage_class,
            })
        }
    }

    fn parse_param_list(&mut self) -> Result<Vec<Spanned<EcoString>>, Error> {
        if self.expect(Token::Void).is_ok() {
            return Ok(Vec::new());
        }

        let mut params = Vec::new();
        loop {
            self.expect(Token::Int)?;
            let ident = self.expect_ident()?;
            params.push(ident);
            if self.expect(Token::Comma).is_err() {
                break;
            }
        }

        Ok(params)
    }

    fn parse_fun_decl(&mut self) -> Result<FunDecl, Error> {
        let storage_class = self.parse_specifiers()?;
        let name = self.expect_ident()?;
        self.expect(Token::OpenParen)?;
        let params = self.parse_param_list()?;
        self.expect(Token::CloseParen)?;
        let body = if self.expect(Token::SemiColon).is_ok() {
            None
        } else {
            Some(self.expect_block()?)
        };
        Ok(FunDecl {
            name,
            params,
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
                Token::Constant(s) => {
                    let value = s.parse()?;
                    let constant = token.clone().map(|_| value);
                    self.advance();
                    Ok(Expression::Constant(constant))
                }
                _ if UnaryOp::try_from(&token.data).is_ok() => {
                    let op = UnaryOp::try_from(&token.data).unwrap();
                    let op = token.clone().map(|_| op);
                    self.advance();
                    let exp = self.parse_factor()?;
                    Ok(Expression::Unary {
                        op,
                        exp: Box::new(exp),
                    })
                }
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
                        })
                    } else {
                        Ok(Expression::Var(Spanned {
                            data: ident,
                            span: span.clone(),
                        }))
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
