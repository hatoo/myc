use std::{fs::File, io::Write, path::PathBuf, process, sync::Arc};

use clap::Parser;

use myc::{
    ast::parse,
    lexer::lexer,
    semantics::{LoopLabel, VarResolver},
    span::SpannedError,
};

#[derive(Debug, Parser)]
struct Opts {
    input: PathBuf,
    #[clap(long)]
    lex: bool,
    #[clap(long)]
    parse: bool,
    #[clap(long)]
    validate: bool,
    #[clap(long)]
    tacky: bool,
    #[clap(long)]
    codegen: bool,
    #[clap(long)]
    asm: bool,
}

fn main() {
    let opts = Opts::parse();
    let src = Arc::new(std::fs::read(&opts.input).unwrap());

    let tokens = lexer(&src)
        .map_err(|err| SpannedError::new(err, src.clone()))
        .unwrap();

    if opts.lex {
        dbg!(tokens);
        return;
    }

    let mut program = parse(&tokens)
        .map_err(|err| SpannedError::new(err, src.clone()))
        .unwrap();

    if opts.parse {
        dbg!(program);
        return;
    }

    VarResolver::default()
        .resolve_program(&mut program)
        .map_err(|err| SpannedError::new(err, src.clone()))
        .unwrap();

    LoopLabel::default()
        .label_program(&mut program)
        .map_err(|err| SpannedError::new(err, src.clone()))
        .unwrap();

    if opts.validate {
        dbg!(program);
        return;
    }

    let tacky = myc::tacky::gen_program(&program);

    if opts.tacky {
        dbg!(tacky);
        return;
    }

    let code = myc::codegen::gen_program(&tacky);

    if opts.codegen {
        dbg!(code);
        return;
    }

    if opts.asm {
        println!("{}", code);
        return;
    }

    File::create(opts.input.with_extension("s"))
        .unwrap()
        .write_all(code.to_string().as_bytes())
        .unwrap();

    process::Command::new("gcc")
        .arg(opts.input.with_extension("s"))
        .arg("-o")
        .arg(opts.input.with_extension(""))
        .status()
        .unwrap();
}
