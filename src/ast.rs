use ecow::EcoString;

use crate::lexer::{Spanned, Token};

#[derive(Debug)]
pub struct Program {
    pub function_definition: Function,
}

#[derive(Debug)]
pub struct Function {
    pub name: EcoString,
    pub body: Statement,
}

#[derive(Debug)]
pub enum Statement {
    Return(Expression),
}

#[derive(Debug)]
pub enum Expression {
    Constant(i32),
    Unary(Unary),
}

#[derive(Debug)]
pub struct Unary {
    pub op: UnaryOp,
    pub exp: Box<Expression>,
}

#[derive(Debug)]
pub enum UnaryOp {
    Complement,
    Negate,
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
    MalfoldExpression(Spanned<Token>),
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
        let body = self.parse_statement()?;
        self.expect(Token::CloseBrace)?;
        Ok(Function {
            name: name.data,
            body,
        })
    }

    fn parse_statement(&mut self) -> Result<Statement, Error> {
        self.expect(Token::Return)?;
        let expr = self.parse_expression()?;
        self.expect(Token::SemiColon)?;
        Ok(Statement::Return(expr))
    }

    fn parse_expression(&mut self) -> Result<Expression, Error> {
        if let Some(token) = self.peek() {
            match &token.data {
                Token::Constant(s) => {
                    let value = s.parse()?;
                    self.advance();
                    Ok(Expression::Constant(value))
                }
                Token::Hyphen => {
                    self.advance();
                    let exp = self.parse_expression()?;
                    Ok(Expression::Unary(Unary {
                        op: UnaryOp::Negate,
                        exp: Box::new(exp),
                    }))
                }
                Token::Tilde => {
                    self.advance();
                    let exp = self.parse_expression()?;
                    Ok(Expression::Unary(Unary {
                        op: UnaryOp::Complement,
                        exp: Box::new(exp),
                    }))
                }
                Token::OpenParen => {
                    self.advance();
                    let exp = self.parse_expression()?;
                    self.expect(Token::CloseParen)?;
                    Ok(exp)
                }
                _ => Err(Error::MalfoldExpression(token.clone())),
            }
        } else {
            Err(Error::UnexpectedEof)
        }
    }
}
