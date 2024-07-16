use std::path::PathBuf;

use clap::Parser;

use myc::lexer::lexer;

#[derive(Debug, Parser)]
struct Opts {
    input: PathBuf,
    #[clap(long)]
    lex: bool,
}

fn main() {
    let opts = Opts::parse();
    let src = std::fs::read(&opts.input).unwrap();
    match lexer(&src) {
        Ok(tokens) => {
            dbg!(tokens);
        }
        Err(err) => {
            err.pretty_print(&src);
            std::process::exit(1);
        }
    }
}
