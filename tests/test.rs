use std::path::Path;

use assert_cmd::Command;

fn test_compile_myc(path: &Path, optimize: bool) {
    let mut command = Command::cargo_bin("myc").unwrap();
    command.arg(path);

    if optimize {
        command.arg("-O");
    }

    command.assert().success();
}

fn assert_myc(path: &Path, return_code: i32, stdout: &[u8]) {
    for optimize in &[false, true] {
        test_compile_myc(path, *optimize);

        let output = std::process::Command::new(path.with_extension(""))
            .output()
            .unwrap();

        assert_eq!(
            output.status.code(),
            Some(return_code),
            "path = {} optimize={}",
            path.to_string_lossy(),
            optimize
        );
        assert_eq!(
            output.stdout.as_slice(),
            stdout,
            "path = {} optimize={}",
            path.to_string_lossy(),
            optimize
        );
    }
}

fn run_gcc(path: &Path) -> (i32, Vec<u8>) {
    std::process::Command::new("gcc")
        .arg(path)
        .arg("-o")
        .arg(path.with_extension(""))
        .output()
        .unwrap();

    let output = std::process::Command::new(path.with_extension(""))
        .output()
        .unwrap();

    (output.status.code().unwrap(), output.stdout)
}

fn clean(path: &Path) {
    let _ = std::fs::remove_file(path.with_extension(""));
    let _ = std::fs::remove_file(path.with_extension("s"));
}

#[test]
fn test_compile_and_run() {
    let dir = Path::new("tests/c/");

    for entry in std::fs::read_dir(dir).unwrap() {
        let entry = entry.unwrap();
        let path = entry.path();

        if path.extension().unwrap() == "c" {
            let (return_code, stdout) = run_gcc(&path);
            assert_myc(&path, return_code, &stdout);
        }
        eprintln!("path = {:?} ok", path);
        clean(&path);
    }
}
