use microlisp::Engine;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("usage: microlisp <file.lisp>");
        eprintln!();
        eprintln!("env vars:");
        eprintln!("  MICROLISP_GC_STRESS=1   collect between every top-level form");
        std::process::exit(1);
    }
    let src = std::fs::read_to_string(&args[1]).expect("read source");

    let mut engine = Engine::new();
    if std::env::var("MICROLISP_GC_STRESS").is_ok() {
        engine.set_gc_stress(true);
        eprintln!("[microlisp] gc-stress mode: collecting between every top-level form");
    }
    let result = engine.run_source(&src);
    println!("=> {}", engine.print(result));
}
