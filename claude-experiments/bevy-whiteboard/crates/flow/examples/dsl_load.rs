//! Load a `.flow` file and either run it to completion OR drop into
//! the REPL. Default path: `flow/examples/dsl_compound_pool.flow`.
//!
//! Usage:
//!   cargo run -p flow --example dsl_load
//!   cargo run -p flow --example dsl_load -- path/to/other.flow
//!   cargo run -p flow --example dsl_load -- path/to.flow --repl

use flow::repl::Repl;

fn main() -> std::io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    // First non-flag argument is the path; flags can appear anywhere.
    let path = args.iter().skip(1)
        .find(|a| !a.starts_with("--"))
        .cloned()
        .unwrap_or_else(|| "flow/examples/dsl_compound_pool.flow".into());
    let want_repl = args.iter().any(|a| a == "--repl");

    let src = std::fs::read_to_string(&path)?;
    let mut sim = match flow::dsl::load(&src, 11) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("{}: parse/lower error: {}", path, e);
            std::process::exit(1);
        }
    };

    println!("Loaded `{}` — {} nodes ({} compound), {} edges, {} params.",
        path,
        sim.nodes.len(),
        sim.nodes.values().filter(|n| n.is_compound()).count(),
        sim.edges.len(),
        sim.params.len());

    if want_repl {
        Repl::new(sim).run();
    } else {
        sim.run_until(500_000_000);
        let completed = sim.log.events.iter().filter(|e| matches!(
            e, flow::Event::MetricRecorded { name, .. } if name == "completed"
        )).count();
        println!("Ran 500ms. Completions recorded: {}", completed);
        println!("(pass --repl to open an interactive session)");
    }
    Ok(())
}
