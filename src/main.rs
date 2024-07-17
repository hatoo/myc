use std::path::PathBuf;

use clap::Parser;

use myc::{ast::parse, lexer::lexer};

#[derive(Debug, Parser)]
struct Opts {
    input: PathBuf,
    #[clap(long)]
    lex: bool,
    #[clap(long)]
    parse: bool,
}

fn main() {
    let opts = Opts::parse();
    let src = std::fs::read(&opts.input).unwrap();

    let tokens = match lexer(&src) {
        Ok(tokens) => tokens,
        Err(err) => {
            err.pretty_print(&src);
            std::process::exit(1);
        }
    };

    if opts.lex {
        dbg!(tokens);
        return;
    }

    let program = parse(&tokens).unwrap();
    dbg!(program);
}
