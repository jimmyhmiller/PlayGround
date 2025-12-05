use iongraph_rust_redux::graph::{Graph, GraphOptions};
use iongraph_rust_redux::compilers::ion::schema::{IonJSON, Pass};  // Use new Ion module
use iongraph_rust_redux::compilers::universal::pass_to_universal;
use iongraph_rust_redux::pure_svg_text_layout_provider::PureSVGTextLayoutProvider;
use std::env;
use std::fs;
use std::process;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!(
            "Usage: {} <input.json> [iterations] [function-index]",
            args[0]
        );
        eprintln!();
        eprintln!("This binary is optimized for profiling with samply:");
        eprintln!(
            "  samply record target/release/profile-render ion-examples/mega-complex.json 1000"
        );
        eprintln!();
        eprintln!("Options:");
        eprintln!("  iterations: Number of times to render (default: 100)");
        eprintln!("  function-index: Index of function to render, or 'all' (default: all)");
        process::exit(1);
    }

    let input_file = &args[1];
    let iterations = args
        .get(2)
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(100);
    let function_index = args.get(3).map(|s| s.as_str());

    // Load the JSON file
    let json_str = fs::read_to_string(input_file).unwrap_or_else(|e| {
        eprintln!("Failed to read {}: {}", input_file, e);
        process::exit(1);
    });

    let ion_json: IonJSON = serde_json::from_str(&json_str).unwrap_or_else(|e| {
        eprintln!("Failed to parse JSON: {}", e);
        process::exit(1);
    });

    println!(
        "Loaded {} functions from {}",
        ion_json.functions.len(),
        input_file
    );
    println!("Running {} iterations...", iterations);
    println!();

    match function_index {
        Some("all") | None => {
            // Render all functions
            for (idx, func) in ion_json.functions.iter().enumerate() {
                for (pass_idx, pass) in func.passes.iter().enumerate() {
                    let block_count = pass
                        .mir
                        .as_ref()
                        .map(|m| m.blocks.len())
                        .or_else(|| pass.lir.as_ref().map(|l| l.blocks.len()))
                        .unwrap_or(0);
                    println!(
                        "Function {}: {} - Pass {}: {} ({} blocks)",
                        idx, func.name, pass_idx, pass.name, block_count
                    );
                    render_pass(pass.clone(), iterations);
                }
            }
        }
        Some(idx_str) => {
            // Render specific function
            let idx = idx_str.parse::<usize>().unwrap_or_else(|_| {
                eprintln!("Invalid function index: {}", idx_str);
                process::exit(1);
            });

            if idx >= ion_json.functions.len() {
                eprintln!(
                    "Function index {} out of range (max: {})",
                    idx,
                    ion_json.functions.len() - 1
                );
                process::exit(1);
            }

            let func = &ion_json.functions[idx];
            for (pass_idx, pass) in func.passes.iter().enumerate() {
                let block_count = pass
                    .mir
                    .as_ref()
                    .map(|m| m.blocks.len())
                    .or_else(|| pass.lir.as_ref().map(|l| l.blocks.len()))
                    .unwrap_or(0);
                println!(
                    "Function {}: {} - Pass {}: {} ({} blocks)",
                    idx, func.name, pass_idx, pass.name, block_count
                );
                render_pass(pass.clone(), iterations);
            }
        }
    }

    println!("\nDone!");
}

fn render_pass(pass: Pass, iterations: usize) {
    // Convert to Universal IR once (not in the loop)
    let universal_ir = pass_to_universal(&pass, "benchmark");

    let options = GraphOptions {
        sample_counts: None,
        instruction_palette: None,
    };

    // Warmup
    for _ in 0..10 {
        let layout_provider = PureSVGTextLayoutProvider::new();
        let mut graph = Graph::new(layout_provider, universal_ir.clone(), options.clone());
        let (nodes_by_layer, layer_heights, track_heights) = graph.layout();
        graph.render(nodes_by_layer, layer_heights, track_heights);
    }

    // Actual iterations for profiling
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        let layout_provider = PureSVGTextLayoutProvider::new();
        let mut graph = Graph::new(layout_provider, universal_ir.clone(), options.clone());
        let (nodes_by_layer, layer_heights, track_heights) = graph.layout();
        graph.render(nodes_by_layer, layer_heights, track_heights);
    }
    let elapsed = start.elapsed();

    let avg_ms = elapsed.as_secs_f64() * 1000.0 / iterations as f64;
    println!("  Average: {:.3} ms per render", avg_ms);
    println!(
        "  Total:   {:.3} ms ({} iterations)",
        elapsed.as_secs_f64() * 1000.0,
        iterations
    );
}
