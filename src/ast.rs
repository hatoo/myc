use ecow::EcoString;

use crate::lexer::{Spanned, Token};

pub struct Program {
    pub function_definition: Function,
}

pub struct Function {
    pub name: EcoString,
    pub body: Statement,
}

pub enum Statement {
    Return(Expression),
}

pub enum Expression {
    Constant(i32),
}

struct Parser {
    tokens: Vec<Spanned<Token>>,
    index: usize,
}

#[derive(Debug)]
pub enum ExpectedToken {
    Token(Token),
    Ident,
    Constant,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Unexpected token: {0:?}, expected {1:?}")]
    Unexpected(Spanned<Token>, ExpectedToken),
    #[error("Unexpected Eof")]
    UnexpectedEof,
    #[error(transparent)]
    ParseIntError(#[from] std::num::ParseIntError),
}

impl Parser {
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
                return Ok(Spanned {
                    data: t.clone(),
                    span: spanned.span.clone(),
                });
            } else {
                return Err(Error::Unexpected(spanned.clone(), ExpectedToken::Ident));
            }
        } else {
            return Err(Error::UnexpectedEof);
        }
    }

    fn expect_constant(&mut self) -> Result<Spanned<EcoString>, Error> {
        if let Some(spanned) = self.tokens.get(self.index) {
            if let Token::Constant(t) = &spanned.data {
                self.index += 1;
                return Ok(Spanned {
                    data: t.clone(),
                    span: spanned.span.clone(),
                });
            } else {
                return Err(Error::Unexpected(spanned.clone(), ExpectedToken::Ident));
            }
        } else {
            return Err(Error::UnexpectedEof);
        }
    }

    fn parse_program(&mut self) -> Result<Program, Error> {
        Ok(Program {
            function_definition: self.parse_function()?,
        })
    }

    fn parse_function(&mut self) -> Result<Function, Error> {
        self.expect(Token::Int)?;
        let name = self.expect_ident()?;
        self.expect(Token::OpenParen)?;
        self.expect(Token::Ident("void".into()))?;
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
        Ok(Statement::Return(expr))
    }

    fn parse_expression(&mut self) -> Result<Expression, Error> {
        let constant = self.expect_constant()?;
        Ok(Expression::Constant(constant.data.parse()?))
    }
}
