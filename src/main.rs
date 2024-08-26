use std::{
    fs::File,
    io::{stdin, Read, Write},
    path::PathBuf,
    process,
    sync::Arc,
};

use clap::Parser;

use myc::{
    ast::parse,
    codegen::CodeGen,
    lexer::{lexer, TokenSpannedError},
    semantics::{LoopLabel, TypeChecker, VarResolver},
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
    #[clap(short)]
    compile: bool,
    #[clap(short)]
    l: Vec<String>,
}

fn main() {
    let opts = Opts::parse();

    let src = if opts.input == PathBuf::from("-") {
        let mut src = Vec::new();
        stdin().read_to_end(&mut src).unwrap();

        // TODO implement a proper preprocessor
        let mut preped = process::Command::new("gcc")
            .arg("-E")
            .arg("-P")
            .arg("-")
            .stdin(process::Stdio::piped())
            .stdout(process::Stdio::piped())
            .spawn()
            .unwrap();

        preped.stdin.take().unwrap().write_all(&src).unwrap();
        preped.wait_with_output().unwrap().stdout
    } else {
        let preped = process::Command::new("gcc")
            .arg("-E")
            .arg("-P")
            .arg(&opts.input)
            .stdout(process::Stdio::piped())
            .spawn()
            .unwrap();

        preped.wait_with_output().unwrap().stdout
    };

    let src = Arc::new(src);

    let tokens = lexer(&src)
        .map_err(|err| SpannedError::new(err, src.clone()))
        .unwrap();

    if opts.lex {
        dbg!(tokens);
        return;
    }

    let mut program = parse(&tokens).unwrap();

    if opts.parse {
        dbg!(program);
        return;
    }

    VarResolver::default()
        .resolve_program(&mut program)
        .map_err(|e| TokenSpannedError {
            error: e,
            src: src.clone(),
            tokens: &tokens,
        })
        .unwrap();

    LoopLabel::default()
        .label_program(&mut program)
        .map_err(|e| TokenSpannedError {
            error: e,
            src: src.clone(),
            tokens: &tokens,
        })
        .unwrap();

    let mut type_checker = TypeChecker::default();
    type_checker
        .check_program(&mut program)
        .map_err(|e| TokenSpannedError {
            error: e,
            src: src.clone(),
            tokens: &tokens,
        })
        .unwrap();

    if opts.validate {
        dbg!(program);
        return;
    }

    let tacky = myc::tacky::gen_program(&program, &mut type_checker.sym_table);

    if opts.tacky {
        for item in tacky.top_levels {
            match item {
                myc::tacky::TopLevelItem::Function(f) => {
                    println!(
                        "{} {} {:?}:",
                        if f.global { "global" } else { "private" },
                        f.name,
                        f.params
                    );

                    for inst in f.body {
                        println!("    {:?}", inst);
                    }
                }
                myc::tacky::TopLevelItem::StaticVariable(v) => {
                    println!(
                        "static {} {} align({}) = {:?};",
                        if v.global { "global" } else { "private" },
                        v.name,
                        v.alignment,
                        v.init
                    );
                }
                myc::tacky::TopLevelItem::StaticConstant(s) => {
                    println!("const {} = {:?}", s.name, s.init)
                }
            }
        }
        return;
    }

    let mut codegen = CodeGen::new(&type_checker.sym_table);
    let code = codegen.gen_program(&tacky);

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

    if opts.compile {
        let mut command = process::Command::new("gcc");
        command
            .arg("-c")
            .arg(opts.input.with_extension("s"))
            .arg("-o")
            .arg(opts.input.with_extension("o"));
        for lib in opts.l {
            command.arg(format!("-l{}", lib));
        }
        command.status().unwrap();
    } else {
        let mut command = process::Command::new("gcc");
        command
            .arg(opts.input.with_extension("s"))
            .arg("-o")
            .arg(opts.input.with_extension(""));
        for lib in opts.l {
            command.arg(format!("-l{}", lib));
        }
        command.status().unwrap();
    }
}
