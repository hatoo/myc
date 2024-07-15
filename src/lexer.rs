use std::ops::Range;

use ecow::EcoString;

#[derive(Debug)]
pub enum Token {
    Ident(EcoString),
    Constant(EcoString),
    Int,
    Void,
    Return,
    OpenParen,
    CloseParen,
    OpenBrace,
    CloseBrace,
    SemiColon,
}

#[derive(Debug)]
pub struct Spanned {
    pub token: Token,
    pub span: Range<usize>,
}

pub fn lexer(src: &[u8]) -> Vec<Spanned> {
    let mut tokens = Vec::new();

    let mut index = 0;

    while index < src.len() {
        // TODO: Support utf-8

        let c = src[index];
        match c {
            _ if c.is_ascii_whitespace() => {
                index += 1;
            }
            b'0'..=b'9' => {
                let start = index;
                while index < src.len() && src[index].is_ascii_digit() {
                    index += 1;
                }
                tokens.push(Spanned {
                    token: Token::Constant(EcoString::from(
                        std::str::from_utf8(&src[start..index]).unwrap(),
                    )),
                    span: start..index,
                });
            }
            _ if c.is_ascii_alphanumeric() || c == b'_' => {
                let start = index;
                while index < src.len() && {
                    let c = src[index];
                    c.is_ascii_alphabetic() || c == b'_'
                } {
                    index += 1;
                }
                let ident = std::str::from_utf8(&src[start..index]).unwrap();
                let token = match ident {
                    "int" => Token::Int,
                    "void" => Token::Void,
                    "return" => Token::Return,
                    _ => Token::Ident(EcoString::from(ident)),
                };
                tokens.push(Spanned {
                    token,
                    span: start..index,
                });
            }
            b';' => {
                tokens.push(Spanned {
                    token: Token::SemiColon,
                    span: index..index + 1,
                });
                index += 1;
            }
            b'(' => {
                tokens.push(Spanned {
                    token: Token::OpenParen,
                    span: index..index + 1,
                });
                index += 1;
            }
            b')' => {
                tokens.push(Spanned {
                    token: Token::CloseParen,
                    span: index..index + 1,
                });
                index += 1;
            }
            b'{' => {
                tokens.push(Spanned {
                    token: Token::OpenBrace,
                    span: index..index + 1,
                });
                index += 1;
            }
            b'}' => {
                tokens.push(Spanned {
                    token: Token::CloseBrace,
                    span: index..index + 1,
                });
                index += 1;
            }

            c => panic!("Unexpected character: {}", c as char),
        }
    }

    tokens
}
