use ecow::EcoString;

use crate::{
    lexer::Token,
    span::{MayHasSpan, Spanned},
};

pub type Identifier = Spanned<EcoString>;

#[derive(Debug)]
pub struct Program {
    pub function_definition: Function,
}

#[derive(Debug)]
pub struct Function {
    pub name: EcoString,
    pub body: Vec<BlockItem>,
}

#[derive(Debug)]
pub enum BlockItem {
    Declaration(Declaration),
    Statement(Statement),
}

#[derive(Debug)]
pub enum Statement {
    Return(Expression),
    Expression(Expression),
    Null,
}

#[derive(Debug)]
pub enum Expression {
    Var(Spanned<EcoString>),
    Constant(i32),
    Unary {
        op: UnaryOp,
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
}

#[derive(Debug)]
pub struct Declaration {
    pub ident: Spanned<EcoString>,
    pub exp: Option<Expression>,
}

#[derive(Debug)]
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

#[derive(Debug)]
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
    fn span(&self) -> Option<std::ops::Range<usize>> {
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
    fn expect(&mut self, token: Token) -> Result<(), Error> {
        if let Some(spanned) = self.tokens.get(self.index) {
            if spanned.data == token {
                self.index += 1;
            } else {
                return Err(Error::Unexpected(
                    spanned.clone(),
                    ExpectedToken::Token(token),
                ));
            }
        } else {
            return Err(Error::UnexpectedEof);
        }
        Ok(())
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

    fn peek(&self) -> Option<&Spanned<Token>> {
        self.tokens.get(self.index)
    }

    fn advance(&mut self) {
        self.index += 1;
        debug_assert!(self.index <= self.tokens.len());
    }

    fn parse_program(&mut self) -> Result<Program, Error> {
        let function_definition = self.parse_function()?;
        self.expect_eof()?;
        Ok(Program {
            function_definition,
        })
    }

    fn parse_function(&mut self) -> Result<Function, Error> {
        self.expect(Token::Int)?;
        let name = self.expect_ident()?;
        self.expect(Token::OpenParen)?;
        self.expect(Token::Void)?;
        self.expect(Token::CloseParen)?;
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
        Ok(Function {
            name: name.data,
            body,
        })
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
            Some(_) => {
                let exp = self.parse_expression(0)?;
                self.expect(Token::SemiColon)?;
                Ok(Statement::Expression(exp))
            }
            None => Err(Error::UnexpectedEof),
        }
    }

    fn parse_declaration(&mut self) -> Result<Declaration, Error> {
        self.expect(Token::Int)?;
        let ident = self.expect_ident()?;
        if self.expect(Token::Equal).is_ok() {
            let exp = self.parse_expression(0)?;
            self.expect(Token::SemiColon)?;
            Ok(Declaration {
                ident,
                exp: Some(exp),
            })
        } else {
            self.expect(Token::SemiColon)?;
            Ok(Declaration { ident, exp: None })
        }
    }

    fn parse_factor(&mut self) -> Result<Expression, Error> {
        if let Some(token) = self.peek() {
            match &token.data {
                Token::Constant(s) => {
                    let value = s.parse()?;
                    self.advance();
                    Ok(Expression::Constant(value))
                }
                _ if UnaryOp::try_from(&token.data).is_ok() => {
                    let op = UnaryOp::try_from(&token.data).unwrap();
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
                    let exp = Expression::Var(Spanned {
                        data: ident.clone(),
                        span: token.span.clone(),
                    });
                    self.advance();
                    Ok(exp)
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
            }

            impl Op {
                fn precedence(&self) -> usize {
                    match self {
                        Self::Binary(op) => op.precedence(),
                        Self::Assign => 1,
                    }
                }
            }

            let op = match token.data {
                Token::Equal => Op::Assign,
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
