// Convert Ion JSON to DOT format for layout comparison
//
// Usage:
//   ion_to_dot <ion-json> <function-index> [pass-index] [output.dot]

use iongraph_rust_redux::compilers::ion::schema::IonJSON;
use iongraph_rust_redux::compilers::universal::pass_to_universal;
use std::env;
use std::fs;
use std::io::Write;
use std::process;

fn print_usage(program_name: &str) {
    eprintln!("Usage:");
    eprintln!("  {} <ion-json> <function-index> [pass-index] [output.dot]", program_name);
    eprintln!();
    eprintln!("Converts Ion JSON to DOT format for layout comparison.");
    eprintln!();
    eprintln!("Examples:");
    eprintln!("  {} mega-complex.json 5 0 output.dot", program_name);
    eprintln!("  {} mega-complex.json 5 > output.dot", program_name);
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let program_name = &args[0];

    if args.len() < 3 {
        print_usage(program_name);
        process::exit(1);
    }

    let json_path = &args[1];
    let func_idx: usize = args[2].parse().unwrap_or_else(|_| {
        eprintln!("Error: function-index must be a number");
        process::exit(1);
    });
    let pass_idx: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(0);
    let output_path = args.get(4);

    // Read and parse the JSON file
    let json_str = fs::read_to_string(json_path).unwrap_or_else(|err| {
        eprintln!("Error reading file {}: {}", json_path, err);
        process::exit(1);
    });

    let data: IonJSON = iongraph_rust_redux::json_compat::parse_as(&json_str).unwrap_or_else(|err| {
        eprintln!("Error parsing Ion JSON: {}", err);
        process::exit(1);
    });

    // Get the specified function and pass
    if func_idx >= data.functions.len() {
        eprintln!(
            "Error: function index {} out of range (max: {})",
            func_idx,
            data.functions.len() - 1
        );
        process::exit(1);
    }

    let func = &data.functions[func_idx];

    if pass_idx >= func.passes.len() {
        eprintln!(
            "Error: pass index {} out of range for function {} (max: {})",
            pass_idx,
            func_idx,
            func.passes.len() - 1
        );
        process::exit(1);
    }

    let pass = &func.passes[pass_idx];
    let func_name = &func.name;
    let pass_name = &pass.name;

    eprintln!(
        "Converting Ion function {} \"{}\", pass {}: \"{}\"",
        func_idx, func_name, pass_idx, pass_name
    );

    // Convert to Universal IR to get the block structure
    let universal_ir = pass_to_universal(pass, func_name);

    eprintln!("  Blocks: {}", universal_ir.blocks.len());

    // Generate DOT output
    let dot_output = generate_dot(&universal_ir, func_name, pass_name);

    // Write output
    if let Some(path) = output_path {
        fs::write(path, &dot_output).unwrap_or_else(|err| {
            eprintln!("Error writing to {}: {}", path, err);
            process::exit(1);
        });
        eprintln!("✓ DOT file generated: {}", path);
    } else {
        // Write to stdout
        std::io::stdout().write_all(dot_output.as_bytes()).unwrap();
    }
}

fn generate_dot(
    ir: &iongraph_rust_redux::compilers::universal::UniversalIR,
    func_name: &str,
    pass_name: &str,
) -> String {
    let mut output = String::new();

    // Sanitize graph name (replace non-alphanumeric with underscore)
    let graph_name: String = func_name
        .chars()
        .map(|c| if c.is_alphanumeric() { c } else { '_' })
        .collect();

    output.push_str(&format!(
        "// Converted from Ion JSON: {} - {}\n",
        func_name, pass_name
    ));
    output.push_str(&format!("digraph {} {{\n", graph_name));

    // Generate nodes
    for block in &ir.blocks {
        let mut attrs = Vec::new();

        // Build label from instructions
        let label: String = block
            .instructions
            .iter()
            .map(|ins| ins.opcode.replace('"', "\\\""))
            .collect::<Vec<_>>()
            .join("\\n");

        attrs.push(format!("label=\"{}\"", label));

        // Add loop attributes
        if block.attributes.iter().any(|a| a == "loopheader" || a == "loop.header") {
            attrs.push("loopheader=\"true\"".to_string());
        }
        if block.attributes.iter().any(|a| a == "backedge" || a == "loop.latch") {
            attrs.push("backedge=\"true\"".to_string());
        }

        output.push_str(&format!("    {} [{}]\n", block.id, attrs.join(" ")));
    }

    output.push('\n');

    // Generate edges
    for block in &ir.blocks {
        for successor in &block.successors {
            output.push_str(&format!("    {} -> {}\n", block.id, successor));
        }
    }

    output.push_str("}\n");

    output
}
