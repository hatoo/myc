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

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Unexpected token: {0:?}, expected {1:?}")]
    Unexpected(Token, Spanned<Token>),
    #[error("Unexpected token: {0:?}, expected Eof")]
    UnexpectedEof(Token),
}

impl Parser {
    fn expect(&mut self, token: Token) -> Result<(), Error> {
        if let Some(spanned) = self.tokens.get(self.index) {
            if spanned.data == token {
                self.index += 1;
            } else {
                return Err(Error::Unexpected(token, spanned.clone()));
            }
        } else {
            return Err(Error::UnexpectedEof(token));
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
                return Err(Error::Unexpected(Token::Ident("".into()), spanned.clone()));
            }
        } else {
            return Err(Error::UnexpectedEof(Token::Ident("".into())));
        }
    }

    fn parse_program(&mut self) -> Result<Program, Error> {
        Ok(Program {
            function_definition: self.parse_function()?,
        })
    }

    fn parse_function(&mut self) -> Result<Function, Error> {
        self.expect(Token::Int);
        self.expect(Token::Ident("main".into()));
        self.expect(Token::OpenParen);
        self.expect(Token::CloseParen);
        self.expect(Token::OpenBrace);
        let body = self.parse_statement();
        self.expect(Token::CloseBrace);
        Function {
            name: "main".into(),
            body,
        }
    }
}
