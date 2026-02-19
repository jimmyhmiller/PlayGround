use std::time::Instant;
use rules4::*;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: rules4 <file>");
        std::process::exit(1);
    }

    let filename = &args[1];
    let src = std::fs::read_to_string(filename).unwrap_or_else(|e| {
        eprintln!("Error reading {}: {}", filename, e);
        std::process::exit(1);
    });

    let mut store = TermStore::new();
    let tokens = Lexer::new(&src).tokenize();
    let program = Parser::new(tokens, &mut store).parse_program();

    let term = pattern_to_term(&mut store, &program.expr);

    let mut engine = Engine::new(store, program.rules, program.meta_rules);

    let start = Instant::now();
    let result = engine.eval(term);
    let elapsed = start.elapsed();

    println!("{}", engine.store.display(result));
    eprintln!("{} reductions in {:.3}s", engine.step_count, elapsed.as_secs_f64());
}
