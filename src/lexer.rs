use ecow::EcoString;
use regex::bytes::Regex;

use crate::span::{MayHasSpan, Spanned};

#[derive(Debug, PartialEq, Clone)]
pub enum Constant {
    Integer { value: u64, suffix: Suffix },
    Float(f64),
}

#[derive(Debug, PartialEq, Clone)]
pub enum Token {
    Ident(EcoString),
    Constant(Constant),
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
    Long,
    Signed,
    Unsigned,
    Double,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Suffix {
    pub l: bool,
    pub u: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Unexpected character: {0:?}")]
    Unexpected(Spanned<char>),
    #[error("Failed to parse integer: {0:?}")]
    ParseIntError(Spanned<std::num::ParseIntError>),
}

impl MayHasSpan for Error {
    fn may_span(&self) -> Option<std::ops::Range<usize>> {
        match self {
            Error::Unexpected(spanned) => Some(spanned.span.clone()),
            Error::ParseIntError(spanned) => Some(spanned.span.clone()),
        }
    }
}

pub fn lexer(src: &[u8]) -> Result<Vec<Spanned<Token>>, Error> {
    let mut tokens = Vec::new();

    let float_re =
        Regex::new(r"^(([0-9]*\.[0-9]+|[0-9]+\.?)[Ee][+-]?[0-9]+|[0-9]*\.[0-9]+|[0-9]+\.)")
            .unwrap();

    let mut index = 0;

    while index < src.len() {
        // TODO: Support utf-8

        let c = src[index];
        match c {
            _ if c.is_ascii_whitespace() => {
                index += 1;
            }
            b'0'..=b'9' => {
                // float
                if let Some(m) = float_re.find(&src[index..]) {
                    debug_assert_eq!(m.start(), 0);
                    tokens.push(Spanned {
                        data: Token::Constant(Constant::Float(
                            std::str::from_utf8(&src[index..index + m.end()])
                                .unwrap()
                                .parse()
                                .unwrap(),
                        )),
                        span: index..index + m.end(),
                    });
                    index += m.len();

                    if index < src.len()
                        && (src[index].is_ascii_alphanumeric()
                            || src[index] == b'_'
                            || src[index] == b'.')
                    {
                        return Err(Error::Unexpected(Spanned {
                            data: src[index] as char,
                            span: index..index + 1,
                        }));
                    }
                } else {
                    let start = index;
                    while index < src.len() && src[index].is_ascii_digit() {
                        index += 1;
                    }
                    let value: u64 = std::str::from_utf8(&src[start..index])
                        .unwrap()
                        .parse()
                        .map_err(|err| {
                            Error::ParseIntError(Spanned {
                                data: err,
                                span: start..index,
                            })
                        })?;
                    let mut suffix = Suffix { l: false, u: false };
                    for _ in 0..2 {
                        if index < src.len() && (src[index] == b'l' || src[index] == b'L') {
                            if suffix.l {
                                return Err(Error::Unexpected(Spanned {
                                    data: src[index] as char,
                                    span: index..index + 1,
                                }));
                            }
                            suffix.l = true;
                            index += 1;
                        }
                        if index < src.len() && (src[index] == b'u' || src[index] == b'U') {
                            if suffix.u {
                                return Err(Error::Unexpected(Spanned {
                                    data: src[index] as char,
                                    span: index..index + 1,
                                }));
                            }
                            suffix.u = true;
                            index += 1;
                        }
                    }
                    if index < src.len()
                        && (src[index].is_ascii_alphanumeric() || src[index] == b'_')
                    {
                        return Err(Error::Unexpected(Spanned {
                            data: src[index] as char,
                            span: index..index + 1,
                        }));
                    }

                    tokens.push(Spanned {
                        data: Token::Constant(Constant::Integer { value, suffix }),
                        span: start..index,
                    });
                }
            }
            b'.' => {
                // float
                if let Some(m) = float_re.find(&src[index..]) {
                    debug_assert_eq!(m.start(), 0);
                    tokens.push(Spanned {
                        data: Token::Constant(Constant::Float(
                            std::str::from_utf8(&src[index..index + m.end()])
                                .unwrap()
                                .parse()
                                .unwrap(),
                        )),
                        span: index..index + m.end(),
                    });
                    index += m.len();

                    if index < src.len()
                        && (src[index].is_ascii_alphanumeric()
                            || src[index] == b'_'
                            || src[index] == b'.')
                    {
                        return Err(Error::Unexpected(Spanned {
                            data: src[index] as char,
                            span: index..index + 1,
                        }));
                    }
                } else {
                    return Err(Error::Unexpected(Spanned {
                        data: c as char,
                        span: index..index + 1,
                    }));
                }
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
                    "long" => Token::Long,
                    "signed" => Token::Signed,
                    "unsigned" => Token::Unsigned,
                    "double" => Token::Double,
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
