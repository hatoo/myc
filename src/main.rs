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
    let tokens = lexer(&src);
    println!("{:#?}", tokens);
}
