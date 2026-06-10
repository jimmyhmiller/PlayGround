use funct::{Funct, Value};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    // `funct test [path]` — run funct-written tests (test_* functions)
    if args.get(1).map(|s| s.as_str()) == Some("test") {
        let target = args.get(2).map(String::as_str).unwrap_or(".");
        let ok = funct::testing::run_and_print(std::path::Path::new(target));
        std::process::exit(if ok { 0 } else { 1 });
    }
    let path = match args.as_slice() {
        [_, cmd, path] if cmd == "run" => path.clone(),
        [_, path] => path.clone(),
        _ => {
            eprintln!("usage: funct [run] <file.ft> | funct test [file-or-dir]");
            std::process::exit(2);
        }
    };
    let src = match std::fs::read_to_string(&path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("error: cannot read {}: {}", path, e);
            std::process::exit(2);
        }
    };
    let mut vm = Funct::new();
    // imports resolve relative to the script's directory
    if let Some(dir) = std::path::Path::new(&path).parent() {
        if !dir.as_os_str().is_empty() {
            vm.set_module_root(dir);
        }
    }
    match vm.eval(&src) {
        Ok(Value::Unit) => {}
        Ok(v) => println!("{}", v),
        Err(e) => {
            eprintln!("{}", e);
            std::process::exit(1);
        }
    }
}
