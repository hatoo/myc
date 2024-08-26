use std::{
    fmt::Display,
    ops::Range,
    sync::{Arc, LazyLock},
};

use ecow::EcoString;
use regex::bytes::Regex;

use crate::span::{self, MayHasSpan, Spanned};

#[derive(Debug, Clone, PartialEq)]
pub struct TokenSpanned<T> {
    pub data: T,
    pub span: Range<usize>,
}

impl<T> TokenSpanned<T> {
    pub fn new_null(data: T) -> Self {
        Self { data, span: 0..0 }
    }

    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> TokenSpanned<U> {
        TokenSpanned {
            data: f(self.data),
            span: self.span,
        }
    }
}

impl<T: Display> Display for TokenSpanned<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.data)
    }
}

pub struct TokenSpannedError<'a, E> {
    pub error: E,
    pub src: Arc<Vec<u8>>,
    pub tokens: &'a [span::Spanned<Token>],
}

impl<'a, E: Display + MayHasTokenSpan> std::fmt::Debug for TokenSpannedError<'a, E> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if let Some(span) = self.error.may_token_span() {
            let span = self.tokens[span.start].span.start..self.tokens[span.end - 1].span.end;
            writeln!(f)?;
            write!(
                f,
                "{}",
                span::SpannedError::new(
                    span::Spanned {
                        data: &self.error,
                        span,
                    },
                    self.src.clone()
                )
            )?
        } else {
            todo!()
        }

        Ok(())
    }
}

pub trait HasTokenSpan {
    fn token_span(&self) -> Range<usize>;
}

pub trait MayHasTokenSpan {
    fn may_token_span(&self) -> Option<Range<usize>>;
}

impl<T> MayHasTokenSpan for T
where
    T: HasTokenSpan,
{
    fn may_token_span(&self) -> Option<Range<usize>> {
        Some(self.token_span())
    }
}

#[derive(Debug, PartialEq, Clone)]
pub enum Constant {
    Integer { value: u64, suffix: Suffix },
    Float(f64),
    Char(u8),
    String(Vec<u8>),
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
    OpenSquareBracket,
    CloseSquareBracket,
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
    Ampersands,
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
    Char,
    Sizeof,
    Struct,
    Dot,
    Arrow,
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct Suffix {
    pub l: bool,
    pub u: bool,
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("Unexpected character: {0}")]
    Unexpected(Spanned<char>),
    #[error("Failed to parse integer: {0}")]
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
    static FLOAT_RE: LazyLock<Regex> = LazyLock::new(|| {
        Regex::new(r"^(([0-9]*\.[0-9]+|[0-9]+\.?)[Ee][+-]?[0-9]+|[0-9]*\.[0-9]+|[0-9]+\.)").unwrap()
    });
    static CHAR_RE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r#"^'(([^'\\\n])|(\\['"?\\abfnrtv0]))'"#).unwrap());
    static STRING_RE: LazyLock<Regex> =
        LazyLock::new(|| Regex::new(r#"^"((([^"\\\n])|(\\['"?\\abfnrtv0]))*)""#).unwrap());

    let mut tokens = Vec::new();

    let mut index = 0;

    while index < src.len() {
        let c = src[index];
        match c {
            _ if c.is_ascii_whitespace() => {
                index += 1;
            }
            b'0'..=b'9' => {
                // float
                if let Some(m) = FLOAT_RE.find(&src[index..]) {
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
                if let Some(m) = FLOAT_RE.find(&src[index..]) {
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
                    tokens.push(Spanned {
                        data: Token::Dot,
                        span: index..index + 1,
                    });
                    index += 1;
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
                    "char" => Token::Char,
                    "sizeof" => Token::Sizeof,
                    "struct" => Token::Struct,
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
            b'[' => {
                tokens.push(Spanned {
                    data: Token::OpenSquareBracket,
                    span: index..index + 1,
                });
                index += 1;
            }
            b']' => {
                tokens.push(Spanned {
                    data: Token::CloseSquareBracket,
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
                } else if index < src.len() && src[index] == b'>' {
                    index += 1;
                    tokens.push(Spanned {
                        data: Token::Arrow,
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
                    tokens.push(Spanned {
                        data: Token::Ampersands,
                        span: index - 1..index,
                    });
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
            b'\'' => {
                if let Some(cap) = CHAR_RE.captures(&src[index..]) {
                    let c = if let Some(g) = cap.get(2) {
                        g.as_bytes()[0]
                    } else {
                        match cap.get(3).unwrap().as_bytes()[1] {
                            b'\'' => b'\'',
                            b'?' => b'?',
                            b'\\' => b'\\',
                            b'"' => b'"',
                            b'a' => b'\x07',
                            b'b' => b'\x08',
                            b'f' => b'\x0c',
                            b'n' => b'\n',
                            b'r' => b'\r',
                            b't' => b'\t',
                            b'v' => b'\x0b',
                            b'0' => b'\0',
                            _ => unreachable!(),
                        }
                    };
                    tokens.push(Spanned {
                        data: Token::Constant(Constant::Char(c)),
                        span: index..index + cap[0].len(),
                    });

                    index += cap[0].len();
                } else {
                    return Err(Error::Unexpected(Spanned {
                        data: c as char,
                        span: index..index + 1,
                    }));
                }
            }
            b'"' => {
                if let Some(cap) = STRING_RE.captures(&src[index..]) {
                    let mut s = Vec::new();
                    let mut iter = cap.get(1).unwrap().as_bytes().iter();

                    while let Some(&b) = iter.next() {
                        if b == b'\\' {
                            let c = match iter.next() {
                                Some(&b) => match b {
                                    b'\'' => b'\'',
                                    b'?' => b'?',
                                    b'\\' => b'\\',
                                    b'"' => b'"',
                                    b'a' => b'\x07',
                                    b'b' => b'\x08',
                                    b'f' => b'\x0c',
                                    b'n' => b'\n',
                                    b'r' => b'\r',
                                    b't' => b'\t',
                                    b'v' => b'\x0b',
                                    b'0' => b'\0',
                                    _ => unreachable!(),
                                },
                                None => {
                                    return Err(Error::Unexpected(Spanned {
                                        data: b as char,
                                        span: index + s.len()..index + s.len() + 1,
                                    }))
                                }
                            };
                            s.push(c);
                        } else {
                            s.push(b);
                        }
                    }

                    tokens.push(Spanned {
                        data: Token::Constant(Constant::String(s)),
                        span: index..index + cap[0].len(),
                    });
                    index += cap[0].len();
                } else {
                    return Err(Error::Unexpected(Spanned {
                        data: c as char,
                        span: index..index + 1,
                    }));
                }
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
