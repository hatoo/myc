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

pub trait HasSpan {
    fn span(&self) -> Range<usize>;
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

impl<E> Error for SpannedError<E> where E: Debug + Display + HasSpan {}

impl<E> Debug for SpannedError<E>
where
    E: Display + HasSpan,
{
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        writeln!(f)?;
        pretty_print(f, &self.src, self.error.span())?;
        write!(f, "{}", self.error)?;
        Ok(())
    }
}

impl<E> Display for SpannedError<E>
where
    E: Display + HasSpan,
{
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        pretty_print(f, &self.src, self.error.span())?;
        write!(f, "{}", self.error)?;
        Ok(())
    }
}

pub fn pretty_print(f: &mut Formatter, src: &[u8], span: Range<usize>) -> std::fmt::Result {
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
            last_line_start,
        )
    } else {
        (1, span.start + 1, 0)
    };

    writeln!(
        f,
        "{} line: {}, column: {}",
        std::str::from_utf8(&src[span]).unwrap(),
        ln,
        col
    )?;
    writeln!(
        f,
        "{}",
        String::from_utf8(
            src[last_line_start..]
                .into_iter()
                .copied()
                .take_while(|c| *c != b'\n')
                .collect::<Vec<u8>>()
        )
        .unwrap()
    )?;
    Ok(())
}
