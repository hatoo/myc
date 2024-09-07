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
    control_flow::Cfg,
    lexer::{lexer, TokenSpannedError},
    semantics::{LoopLabel, TypeChecker, VarResolver},
    span::SpannedError,
    ssa::Ssa,
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
    #[clap(long)]
    fold_constants: bool,
    #[clap(long)]
    eliminate_unreachable_code: bool,
    #[clap(long)]
    propagate_copies: bool,
    #[clap(long)]
    eliminate_dead_stores: bool,
    #[clap(short = 'O')]
    optimize: bool,
    #[clap(short = 's')]
    s: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let opts = Opts::parse();

    let src = if opts.input == PathBuf::from("-") {
        let mut src = Vec::new();
        stdin().read_to_end(&mut src)?;

        // TODO implement a proper preprocessor
        let mut preped = process::Command::new("gcc")
            .arg("-E")
            .arg("-P")
            .arg("-")
            .stdin(process::Stdio::piped())
            .stdout(process::Stdio::piped())
            .spawn()?;

        preped.stdin.take().unwrap().write_all(&src)?;
        preped.wait_with_output()?.stdout
    } else {
        let preped = process::Command::new("gcc")
            .arg("-E")
            .arg("-P")
            .arg(&opts.input)
            .stdout(process::Stdio::piped())
            .spawn()?;

        preped.wait_with_output()?.stdout
    };

    let src = Arc::new(src);

    let tokens = lexer(&src).map_err(|err| SpannedError::new(err, src.clone()))?;

    if opts.lex {
        dbg!(tokens);
        return Ok(());
    }

    let tokens = Arc::new(tokens);

    let mut program = parse(&tokens).map_err(|e| TokenSpannedError {
        error: e,
        src: src.clone(),
        tokens: tokens.clone(),
    })?;

    if opts.parse {
        dbg!(program);
        return Ok(());
    }

    VarResolver::default()
        .resolve_program(&mut program)
        .map_err(|e| TokenSpannedError {
            error: e,
            src: src.clone(),
            tokens: tokens.clone(),
        })?;

    LoopLabel::default()
        .label_program(&mut program)
        .map_err(|e| TokenSpannedError {
            error: e,
            src: src.clone(),
            tokens: tokens.clone(),
        })?;

    let mut type_checker = TypeChecker::default();
    type_checker
        .check_program(&mut program)
        .map_err(|e| TokenSpannedError {
            error: e,
            src: src.clone(),
            tokens: tokens.clone(),
        })?;

    if opts.validate {
        dbg!(program);
        return Ok(());
    }

    let mut tacky = myc::tacky::gen_program(&program, &mut type_checker.sym_table);

    for f in &tacky.top_levels {
        if let myc::tacky::TopLevelItem::Function(f) = f {
            let cfg = Cfg::new(&f.body);
            let ssa = Ssa::new(cfg);
            dbg!(ssa);
        }
    }

    if opts.optimize
        || opts.fold_constants
        || opts.eliminate_unreachable_code
        || opts.propagate_copies
        || opts.eliminate_dead_stores
    {
        let mut optimizes = Vec::new();
        if opts.fold_constants || opts.optimize {
            optimizes.push(myc::optimize_tacky::OptimizeOption::ConstantFolding);
        }
        if opts.eliminate_unreachable_code || opts.optimize {
            optimizes.push(myc::optimize_tacky::OptimizeOption::DeadCodeElimination);
        }
        if opts.propagate_copies || opts.optimize {
            optimizes.push(myc::optimize_tacky::OptimizeOption::CopyPropagation);
        }
        if opts.eliminate_dead_stores || opts.optimize {
            optimizes.push(myc::optimize_tacky::OptimizeOption::EliminateDeadStores);
        }
        myc::optimize_tacky::optimize(&mut tacky, &type_checker.sym_table, &optimizes);
    }

    if opts.tacky {
        print_tacky(&tacky);
        return Ok(());
    }

    let mut codegen = CodeGen::new(&type_checker.sym_table);
    let code = codegen.gen_program(&tacky, true);

    if opts.codegen {
        dbg!(code);
        return Ok(());
    }

    if opts.asm {
        println!("{}", code);
        return Ok(());
    }

    if opts.s {
        File::create(opts.input.with_extension("s"))?.write_all(code.to_string().as_bytes())?;
        return Ok(());
    }

    File::create(opts.input.with_extension("s"))?.write_all(code.to_string().as_bytes())?;

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
        command.status()?;
    } else {
        let mut command = process::Command::new("gcc");
        command
            .arg(opts.input.with_extension("s"))
            .arg("-o")
            .arg(opts.input.with_extension(""));
        for lib in opts.l {
            command.arg(format!("-l{}", lib));
        }
        command.status()?;
    }

    Ok(())
}

fn print_tacky(tacky: &myc::tacky::Program) {
    for item in &tacky.top_levels {
        match item {
            myc::tacky::TopLevelItem::Function(f) => {
                println!(
                    "{} {} {:?}:",
                    if f.global { "global" } else { "private" },
                    f.name,
                    f.params
                );

                for inst in &f.body {
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
}
