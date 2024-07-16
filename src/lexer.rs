use std::ops::Range;

use ecow::EcoString;

#[derive(Debug, PartialEq, Eq, Clone)]
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

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Unexpected character: {0:?}")]
    Enexpected(Spanned<char>),
}

impl Error {
    pub fn pretty_print(&self, src: &[u8]) {
        match self {
            Self::Enexpected(span_char) => {
                let (ln, col) = if let Some((line_number, last_line_start)) = src
                    [..span_char.span.start]
                    .iter()
                    .enumerate()
                    .filter(|t| t.1 == &b'\n')
                    .map(|t| t.0)
                    .enumerate()
                    .last()
                {
                    (line_number + 2, span_char.span.start - last_line_start)
                } else {
                    (1, span_char.span.start + 1)
                };

                eprintln!(
                    "Lex Error: Unexpected character: {} at line: {}, column: {}",
                    span_char.data, ln, col
                );
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Spanned<T> {
    pub data: T,
    pub span: Range<usize>,
}

pub fn lexer(src: &[u8]) -> Result<Vec<Spanned<Token>>, Error> {
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
                if index < src.len() && (src[index].is_ascii_alphanumeric() || src[index] == b'_') {
                    return Err(Error::Enexpected(Spanned {
                        data: src[index] as char,
                        span: index..index + 1,
                    }));
                }

                tokens.push(Spanned {
                    data: Token::Constant(EcoString::from(
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
                    data: token,
                    span: start..index,
                });
            }
            b';' => {
                tokens.push(Spanned {
                    data: Token::SemiColon,
                    span: index..index + 1,
                });
                index += 1;
            }
            b'(' => {
                tokens.push(Spanned {
                    data: Token::OpenParen,
                    span: index..index + 1,
                });
                index += 1;
            }
            b')' => {
                tokens.push(Spanned {
                    data: Token::CloseParen,
                    span: index..index + 1,
                });
                index += 1;
            }
            b'{' => {
                tokens.push(Spanned {
                    data: Token::OpenBrace,
                    span: index..index + 1,
                });
                index += 1;
            }
            b'}' => {
                tokens.push(Spanned {
                    data: Token::CloseBrace,
                    span: index..index + 1,
                });
                index += 1;
            }
            b'/' => {
                index += 1;
                if index < src.len() && src[index] == b'/' {
                    while index < src.len() && src[index] != b'\n' {
                        index += 1;
                    }
                } else if index < src.len() && src[index] == b'*' {
                    index += 1;
                    while index < src.len() {
                        if src[index] == b'*' && index + 1 < src.len() && src[index + 1] == b'/' {
                            index += 2;
                            break;
                        }
                        index += 1;
                    }
                } else {
                    return Err(Error::Enexpected(Spanned {
                        data: c as char,
                        span: index..index + 1,
                    }));
                }
            }

            c => {
                return Err(Error::Enexpected(Spanned {
                    data: c as char,
                    span: index..index + 1,
                }));
            }
        }
    }

    Ok(tokens)
}
