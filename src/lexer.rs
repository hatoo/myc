use ecow::EcoString;

use crate::span::{MayHasSpan, Spanned};

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
    Tilde,
    Hyphen,
    TwoHyphens,
    Plus,
    TwoPlus,
    Asterisk,
    Slash,
    Percent,
    Exclamation,
    TwoAmpersands,
    TwoPipes,
    Equal,
    TwoEquals,
    ExclamationEquals,
    LessThan,
    GreaterThan,
    LessThanEquals,
    GreaterThanEquals,
    If,
    Else,
    Question,
    Colon,
    Do,
    While,
    For,
    Break,
    Continue,
    Comma,
    Static,
    Extern,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Unexpected character: {0:?}")]
    Unexpected(Spanned<char>),
}

impl MayHasSpan for Error {
    fn may_span(&self) -> Option<std::ops::Range<usize>> {
        match self {
            Error::Unexpected(spanned) => Some(spanned.span.clone()),
        }
    }
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
                    return Err(Error::Unexpected(Spanned {
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
                    c.is_ascii_alphanumeric() || c == b'_'
                } {
                    index += 1;
                }
                let ident = std::str::from_utf8(&src[start..index]).unwrap();
                let token = match ident {
                    "int" => Token::Int,
                    "void" => Token::Void,
                    "return" => Token::Return,
                    "if" => Token::If,
                    "else" => Token::Else,
                    "do" => Token::Do,
                    "while" => Token::While,
                    "for" => Token::For,
                    "break" => Token::Break,
                    "continue" => Token::Continue,
                    "static" => Token::Static,
                    "extern" => Token::Extern,
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
                    tokens.push(Spanned {
                        data: Token::Slash,
                        span: index - 1..index,
                    });
                }
            }
            b'~' => {
                index += 1;
                tokens.push(Spanned {
                    data: Token::Tilde,
                    span: index..index + 1,
                });
            }
            b'-' => {
                index += 1;
                if index < src.len() && src[index] == b'-' {
                    index += 1;
                    tokens.push(Spanned {
                        data: Token::TwoHyphens,
                        span: index - 2..index,
                    });
                } else {
                    tokens.push(Spanned {
                        data: Token::Hyphen,
                        span: index - 1..index,
                    });
                }
            }
            b'+' => {
                index += 1;
                if index < src.len() && src[index] == b'+' {
                    index += 1;
                    tokens.push(Spanned {
                        data: Token::TwoPlus,
                        span: index - 2..index,
                    });
                } else {
                    tokens.push(Spanned {
                        data: Token::Plus,
                        span: index - 1..index,
                    });
                }
            }
            b'*' => {
                tokens.push(Spanned {
                    data: Token::Asterisk,
                    span: index..index + 1,
                });
                index += 1;
            }
            b'%' => {
                tokens.push(Spanned {
                    data: Token::Percent,
                    span: index..index + 1,
                });
                index += 1;
            }
            b'!' => {
                index += 1;
                if index < src.len() && src[index] == b'=' {
                    index += 1;
                    tokens.push(Spanned {
                        data: Token::ExclamationEquals,
                        span: index - 2..index,
                    });
                } else {
                    tokens.push(Spanned {
                        data: Token::Exclamation,
                        span: index - 1..index,
                    });
                }
            }
            b'&' => {
                index += 1;
                if index < src.len() && src[index] == b'&' {
                    index += 1;
                    tokens.push(Spanned {
                        data: Token::TwoAmpersands,
                        span: index - 2..index,
                    });
                } else {
                    return Err(Error::Unexpected(Spanned {
                        data: src[index] as char,
                        span: index..index + 1,
                    }));
                }
            }
            b'|' => {
                index += 1;
                if index < src.len() && src[index] == b'|' {
                    index += 1;
                    tokens.push(Spanned {
                        data: Token::TwoPipes,
                        span: index - 2..index,
                    });
                } else {
                    return Err(Error::Unexpected(Spanned {
                        data: src[index] as char,
                        span: index..index + 1,
                    }));
                }
            }
            b'=' => {
                index += 1;
                if index < src.len() && src[index] == b'=' {
                    index += 1;
                    tokens.push(Spanned {
                        data: Token::TwoEquals,
                        span: index - 2..index,
                    });
                } else {
                    tokens.push(Spanned {
                        data: Token::Equal,
                        span: index - 1..index,
                    });
                }
            }
            b'<' => {
                index += 1;
                if index < src.len() && src[index] == b'=' {
                    index += 1;
                    tokens.push(Spanned {
                        data: Token::LessThanEquals,
                        span: index - 2..index,
                    });
                } else {
                    tokens.push(Spanned {
                        data: Token::LessThan,
                        span: index - 1..index,
                    });
                }
            }
            b'>' => {
                index += 1;
                if index < src.len() && src[index] == b'=' {
                    index += 1;
                    tokens.push(Spanned {
                        data: Token::GreaterThanEquals,
                        span: index - 2..index,
                    });
                } else {
                    tokens.push(Spanned {
                        data: Token::GreaterThan,
                        span: index - 1..index,
                    });
                }
            }
            b'?' => {
                tokens.push(Spanned {
                    data: Token::Question,
                    span: index..index + 1,
                });
                index += 1;
            }
            b':' => {
                tokens.push(Spanned {
                    data: Token::Colon,
                    span: index..index + 1,
                });
                index += 1;
            }
            b',' => {
                tokens.push(Spanned {
                    data: Token::Comma,
                    span: index..index + 1,
                });
                index += 1;
            }

            // TODO
            b'#' => {
                while index < src.len() && src[index] != b'\n' {
                    index += 1;
                }
            }

            c => {
                return Err(Error::Unexpected(Spanned {
                    data: c as char,
                    span: index..index + 1,
                }));
            }
        }
    }

    Ok(tokens)
}
