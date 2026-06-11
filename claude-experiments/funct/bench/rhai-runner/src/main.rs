// Minimal rhai script runner for benchmarking. `print(x)` -> stdout.
fn main() {
    let path = std::env::args().nth(1).expect("usage: rhai-run <file.rhai>");
    let src = std::fs::read_to_string(&path).expect("read script");
    let mut engine = rhai::Engine::new();
    engine.on_print(|s| println!("{}", s));
    if let Err(e) = engine.run(&src) {
        eprintln!("rhai error: {}", e);
        std::process::exit(1);
    }
}
