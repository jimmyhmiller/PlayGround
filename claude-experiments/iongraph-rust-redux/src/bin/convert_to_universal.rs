use iongraph_rust_redux::compilers::ion::IonJSON;
use iongraph_rust_redux::compilers::universal::{ion_to_universal, pass_to_universal};
use std::env;
use std::fs;
use std::process;

fn main() {
    let args: Vec<String> = env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: {} <ion-json-file> [output-file]", args[0]);
        eprintln!("       {} <ion-json-file> --pass <func-idx> <pass-idx> [output-file]", args[0]);
        eprintln!();
        eprintln!("Examples:");
        eprintln!("  {} input.json output.json           # Convert entire Ion JSON", args[0]);
        eprintln!("  {} input.json --pass 0 5 out.json  # Convert single pass", args[0]);
        process::exit(1);
    }

    let input_file = &args[1];
    let json_str = fs::read_to_string(input_file).unwrap_or_else(|err| {
        eprintln!("Error reading {}: {}", input_file, err);
        process::exit(1);
    });

    let ion: IonJSON = serde_json::from_str(&json_str).unwrap_or_else(|err| {
        eprintln!("Error parsing Ion JSON: {}", err);
        process::exit(1);
    });

    // Check if we're converting a single pass or the entire file
    let universal = if args.len() > 2 && args[2] == "--pass" {
        // Convert single pass
        if args.len() < 5 {
            eprintln!("Error: --pass requires function index and pass index");
            process::exit(1);
        }

        let func_idx: usize = args[3].parse().unwrap_or_else(|_| {
            eprintln!("Error: function index must be a number");
            process::exit(1);
        });

        let pass_idx: usize = args[4].parse().unwrap_or_else(|_| {
            eprintln!("Error: pass index must be a number");
            process::exit(1);
        });

        if func_idx >= ion.functions.len() {
            eprintln!(
                "Error: function index {} out of range (max: {})",
                func_idx,
                ion.functions.len() - 1
            );
            process::exit(1);
        }

        let func = &ion.functions[func_idx];

        if pass_idx >= func.passes.len() {
            eprintln!(
                "Error: pass index {} out of range (max: {})",
                pass_idx,
                func.passes.len() - 1
            );
            process::exit(1);
        }

        let pass = &func.passes[pass_idx];
        eprintln!(
            "Converting function {} \"{}\", pass {} \"{}\"",
            func_idx, func.name, pass_idx, pass.name
        );

        pass_to_universal(pass, &func.name)
    } else {
        // Convert entire file
        eprintln!("Converting entire Ion JSON file...");
        eprintln!("  Functions: {}", ion.functions.len());
        ion_to_universal(&ion)
    };

    // Determine output file
    let output_file = if args.len() > 2 && args[2] == "--pass" {
        args.get(5).map(|s| s.as_str()).unwrap_or("output.json")
    } else {
        args.get(2).map(|s| s.as_str()).unwrap_or("output.json")
    };

    // Serialize to pretty JSON
    let output_json = serde_json::to_string_pretty(&universal).unwrap_or_else(|err| {
        eprintln!("Error serializing to JSON: {}", err);
        process::exit(1);
    });

    fs::write(output_file, output_json).unwrap_or_else(|err| {
        eprintln!("Error writing to {}: {}", output_file, err);
        process::exit(1);
    });

    eprintln!("✓ Universal JSON written to: {}", output_file);
    eprintln!("  Blocks: {}", universal.blocks.len());
}
