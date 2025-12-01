use iongraph_rust::*;
use std::fs;

#[test]
fn test_byte_for_byte_identical_to_typescript() {
    // Path to the TypeScript-generated SVG (the user should copy it here)
    let ts_svg_path = "tests/fixtures/ts-mega-complex-func5-pass0.svg";

    // Read the TypeScript SVG
    let ts_svg = fs::read_to_string(ts_svg_path)
        .expect("TypeScript SVG file not found - please copy the TS output to tests/fixtures/ts-mega-complex-func5-pass0.svg");

    // Generate Rust SVG with same input
    let json_path = "/Users/jimmyhmiller/Documents/Code/open-source/iongraph2/examples/mega-complex.json";
    let json_str = fs::read_to_string(json_path).expect("Failed to read JSON");
    let ion_json: IonJSON = serde_json::from_str(&json_str).expect("Failed to parse JSON");

    let func = &ion_json.functions[5];
    let pass = func.passes[0].clone();

    let mut graph = Graph::new(Vec2::new(5000.0, 5000.0), pass);
    let rust_svg = graph.render_svg();

    // Write Rust output for debugging
    fs::write("tests/fixtures/rust-output.svg", &rust_svg).ok();

    // Compare byte-for-byte
    if ts_svg != rust_svg {
        // Find first difference
        let ts_lines: Vec<&str> = ts_svg.lines().collect();
        let rust_lines: Vec<&str> = rust_svg.lines().collect();

        println!("\n=== DIFFERENCE FOUND ===");
        println!("TypeScript lines: {}", ts_lines.len());
        println!("Rust lines: {}", rust_lines.len());

        for (i, (ts_line, rust_line)) in ts_lines.iter().zip(rust_lines.iter()).enumerate() {
            if ts_line != rust_line {
                println!("\nFirst difference at line {}:", i + 1);
                println!("TS:   {}", ts_line);
                println!("Rust: {}", rust_line);

                // Show character-by-character difference
                for (j, (ts_char, rust_char)) in ts_line.chars().zip(rust_line.chars()).enumerate() {
                    if ts_char != rust_char {
                        println!("  First char diff at column {}: '{}' vs '{}'", j, ts_char, rust_char);
                        break;
                    }
                }
                break;
            }
        }

        if ts_lines.len() != rust_lines.len() {
            println!("\nLine count mismatch!");
            if ts_lines.len() > rust_lines.len() {
                println!("Extra TS lines starting at {}:", rust_lines.len() + 1);
                for line in &ts_lines[rust_lines.len()..ts_lines.len().min(rust_lines.len() + 5)] {
                    println!("  {}", line);
                }
            } else {
                println!("Extra Rust lines starting at {}:", ts_lines.len() + 1);
                for line in &rust_lines[ts_lines.len()..rust_lines.len().min(ts_lines.len() + 5)] {
                    println!("  {}", line);
                }
            }
        }

        panic!("SVG outputs are NOT byte-for-byte identical!");
    }

    println!("✓ SVG outputs are BYTE-FOR-BYTE IDENTICAL!");
}
