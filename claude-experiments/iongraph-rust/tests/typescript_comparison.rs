use iongraph_rust::*;
use std::fs;

/// Test that our Rust implementation produces byte-for-byte identical SVG output
/// to the TypeScript implementation for the fibonacci example
#[test]
fn test_fibonacci_matches_typescript() {
    let ts_svg_path = "tests/fixtures/ts-fibonacci.svg";
    let json_path = "tests/fixtures/fibonacci.json";

    // Read the TypeScript-generated SVG
    let ts_svg = fs::read_to_string(ts_svg_path)
        .expect("TypeScript SVG file not found - run ./generate_test_cases.sh first");

    // Read and parse the JSON
    let json_str = fs::read_to_string(json_path).expect("Failed to read JSON");
    let ion_json: IonJSON = serde_json::from_str(&json_str).expect("Failed to parse JSON");

    // Generate Rust SVG with same input
    let func = &ion_json.functions[0];
    let pass = func.passes[0].clone();

    let mut graph = Graph::new(Vec2::new(5000.0, 5000.0), pass);
    let rust_svg = graph.render_svg();

    // Write Rust output for debugging
    fs::write("tests/fixtures/rust-fibonacci.svg", &rust_svg).ok();

    // Compare byte-for-byte
    compare_svgs(&ts_svg, &rust_svg, "fibonacci");
}

/// Test mega-complex function 5, pass 0
#[test]
fn test_mega_complex_func5_pass0_matches_typescript() {
    let ts_svg_path = "tests/fixtures/ts-mega-complex-func5-pass0.svg";
    let json_path = "tests/fixtures/mega-complex-func5-pass0.json";

    // Read the TypeScript-generated SVG
    let ts_svg = fs::read_to_string(ts_svg_path)
        .expect("TypeScript SVG file not found - run ./generate_test_cases.sh first");

    // Read and parse the JSON
    let json_str = fs::read_to_string(json_path).expect("Failed to read JSON");
    let ion_json: IonJSON = serde_json::from_str(&json_str).expect("Failed to parse JSON");

    // Generate Rust SVG with same input
    let func = &ion_json.functions[5];
    let pass = func.passes[0].clone();

    let mut graph = Graph::new(Vec2::new(5000.0, 5000.0), pass);
    let rust_svg = graph.render_svg();

    // Write Rust output for debugging
    fs::write("tests/fixtures/rust-mega-complex-func5-pass0.svg", &rust_svg).ok();

    // Compare byte-for-byte
    compare_svgs(&ts_svg, &rust_svg, "mega-complex-func5-pass0");
}

/// Test with 50 blocks
#[test]
fn test_50_blocks_matches_typescript() {
    let ts_svg_path = "tests/fixtures/ts-test-50.svg";
    let json_path = "tests/fixtures/test-50.json";

    // Read the TypeScript-generated SVG
    let ts_svg = fs::read_to_string(ts_svg_path)
        .expect("TypeScript SVG file not found - run ./generate_test_cases.sh first");

    // Read and parse the JSON
    let json_str = fs::read_to_string(json_path).expect("Failed to read JSON");
    let ion_json: IonJSON = serde_json::from_str(&json_str).expect("Failed to parse JSON");

    // Generate Rust SVG with same input
    let func = &ion_json.functions[0];
    let pass = func.passes[0].clone();

    let mut graph = Graph::new(Vec2::new(5000.0, 5000.0), pass);
    let rust_svg = graph.render_svg();

    // Write Rust output for debugging
    fs::write("tests/fixtures/rust-test-50.svg", &rust_svg).ok();

    // Compare byte-for-byte
    compare_svgs(&ts_svg, &rust_svg, "test-50");
}

/// Helper function to compare SVGs and provide detailed diff on failure
fn compare_svgs(ts_svg: &str, rust_svg: &str, test_name: &str) {
    if ts_svg == rust_svg {
        println!("✓ {} - SVG outputs are BYTE-FOR-BYTE IDENTICAL!", test_name);
        return;
    }

    // Find first difference
    let ts_lines: Vec<&str> = ts_svg.lines().collect();
    let rust_lines: Vec<&str> = rust_svg.lines().collect();

    println!("\n=== DIFFERENCE FOUND in {} ===", test_name);
    println!("TypeScript lines: {}", ts_lines.len());
    println!("Rust lines: {}", rust_lines.len());

    // Find first differing line
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

            // Show context (previous and next lines)
            if i > 0 {
                println!("\nPrevious line ({}): {}", i, ts_lines[i - 1]);
            }
            if i + 1 < ts_lines.len().min(rust_lines.len()) {
                println!("Next line ({}): {}", i + 2, ts_lines[i + 1]);
            }

            break;
        }
    }

    if ts_lines.len() != rust_lines.len() {
        println!("\n⚠️  Line count mismatch!");
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

    panic!("SVG outputs are NOT byte-for-byte identical for {}!", test_name);
}
