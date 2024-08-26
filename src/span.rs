use std::{
    error::Error,
    fmt::Debug,
    fmt::{Display, Formatter},
    ops::Range,
    sync::Arc,
};

use miette::{LabeledSpan, MietteDiagnostic};

pub type Span = Range<usize>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Spanned<T> {
    pub data: T,
    pub span: Span,
}

impl<T> HasSpan for Spanned<T> {
    fn span(&self) -> Range<usize> {
        self.span.clone()
    }
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
    pub fn new_null(data: T) -> Self {
        Self { data, span: 0..0 }
    }

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
            let report = MietteDiagnostic {
                message: self.error.to_string(),
                code: None,
                severity: None,
                help: None,
                url: None,
                labels: Some(vec![LabeledSpan::new(None, span.start, span.len())]),
            };

            write!(
                f,
                "{:?}",
                miette::Error::new(report).with_source_code(self.src.clone())
            )?;
        }
        Ok(())
    }
}

impl<E> Display for SpannedError<E>
where
    E: Display + MayHasSpan,
{
    fn fmt(&self, f: &mut Formatter) -> std::fmt::Result {
        write!(f, "{:?}", self)?;
        Ok(())
    }
}
