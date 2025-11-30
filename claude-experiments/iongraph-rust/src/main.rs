use iongraph_rust::*;
use std::fs;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 3 {
        eprintln!("Usage: {} <path-to-json> <function-index> [pass-index] [output.svg]", args[0]);
        eprintln!("Example: {} examples/mega-complex.json 5 0 output.svg", args[0]);
        std::process::exit(1);
    }

    let data_path = &args[1];
    let func_index: usize = args[2].parse().expect("Function index must be a number");
    let pass_index: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0);
    let output_path = args.get(4).map(|s| s.as_str()).unwrap_or("output.svg");

    // Read and parse JSON
    println!("Reading {}...", data_path);
    let json_str = fs::read_to_string(data_path).expect("Failed to read JSON file");

    println!("Parsing JSON...");
    let ion_json: IonJSON = serde_json::from_str(&json_str).expect("Failed to parse JSON");

    if ion_json.functions.is_empty() {
        eprintln!("Error: JSON must contain at least one function");
        std::process::exit(1);
    }

    if func_index >= ion_json.functions.len() {
        eprintln!("Error: Function index {} out of range (max: {})", func_index, ion_json.functions.len() - 1);
        std::process::exit(1);
    }

    let func = &ion_json.functions[func_index];
    if func.passes.is_empty() {
        eprintln!("Error: Function must contain at least one pass");
        std::process::exit(1);
    }

    if pass_index >= func.passes.len() {
        eprintln!("Error: Pass index {} out of range (max: {})", pass_index, func.passes.len() - 1);
        std::process::exit(1);
    }

    let pass = func.passes[pass_index].clone();

    println!("Rendering function {} '{}', pass {}: '{}'",
        func_index, func.name, pass_index, pass.name);
    println!("  MIR blocks: {}", pass.mir.blocks.len());

    // Create graph and render SVG with larger viewport for complex graphs
    let mut graph = Graph::new(Vec2::new(5000.0, 5000.0), pass);
    let svg = graph.render_svg();

    // Write output
    fs::write(output_path, &svg).expect("Failed to write SVG file");

    println!("✓ SVG generated: {}", output_path);
    println!("  Size: {:.2} KB", svg.len() as f64 / 1024.0);
}
