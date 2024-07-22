use std::ops::Range;

#[derive(Debug, Clone)]
pub struct Spanned<T> {
    pub data: T,
    pub span: Range<usize>,
}

pub fn pretty_print(src: &[u8], span: Range<usize>) {
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

    eprintln!(
        "{} line: {}, column: {}",
        std::str::from_utf8(&src[span]).unwrap(),
        ln,
        col
    );
    eprintln!(
        "{}",
        String::from_utf8(
            src[last_line_start..]
                .into_iter()
                .copied()
                .take_while(|c| *c != b'\n')
                .collect::<Vec<u8>>()
        )
        .unwrap()
    );
}
