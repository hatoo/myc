use std::{
    error::Error,
    fmt::Debug,
    fmt::{Display, Formatter},
    ops::Range,
    sync::Arc,
};

#[derive(Debug, Clone)]
pub struct Spanned<T> {
    pub data: T,
    pub span: Range<usize>,
}

impl<T> Display for Spanned<T>
where
    T: Display,
{
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "{}", self.data)
    }
}

impl<T> Spanned<T> {
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> Spanned<U> {
        Spanned {
            data: f(self.data),
            span: self.span,
        }
    }
}

pub trait HasSpan {
    fn span(&self) -> Range<usize>;
}

pub trait MayHasSpan {
    fn may_span(&self) -> Option<Range<usize>>;
}

impl<T> MayHasSpan for T
where
    T: HasSpan,
{
    fn may_span(&self) -> Option<Range<usize>> {
        Some(self.span())
    }
}

pub struct SpannedError<E> {
    pub error: E,
    pub src: Arc<Vec<u8>>,
}

impl<E> SpannedError<E> {
    pub fn new(error: E, src: Arc<Vec<u8>>) -> Self {
        Self { error, src }
    }
}

impl<E> Error for SpannedError<E> where E: Debug + Display + MayHasSpan {}

impl<E> Debug for SpannedError<E>
where
    E: Display + MayHasSpan,
{
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        writeln!(f)?;
        if let Some(span) = self.error.may_span() {
            pretty_print(f, &self.src, span, &self.error)?;
        }
        Ok(())
    }
}

impl<E> Display for SpannedError<E>
where
    E: Display + MayHasSpan,
{
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        if let Some(span) = self.error.may_span() {
            pretty_print(f, &self.src, span, &self.error)?;
        }
        Ok(())
    }
}

pub fn pretty_print<E: Display>(
    f: &mut Formatter,
    src: &[u8],
    span: Range<usize>,
    error: E,
) -> std::fmt::Result {
    let (ln, col, last_line_start) = if let Some((line_number, last_line_start)) = src[..span.start]
        .iter()
        .enumerate()
        .filter(|t| t.1 == &b'\n')
        .map(|t| t.0)
        .enumerate()
        .last()
    {
        (
            line_number + 2,
            span.start - last_line_start,
            last_line_start + 1,
        )
    } else {
        (1, span.start + 1, 0)
    };

    writeln!(f, "{}:{} {}", ln, col, error)?;
    writeln!(
        f,
        "{}",
        String::from_utf8(
            src[last_line_start..]
                .iter()
                .copied()
                .take_while(|c| *c != b'\n')
                .collect::<Vec<u8>>()
        )
        .unwrap()
    )?;
    for _ in 1..col {
        write!(f, " ")?;
    }
    for _ in span {
        write!(f, "^")?;
    }
    writeln!(f)?;
    Ok(())
}
