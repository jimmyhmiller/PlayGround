use iongraph_rust::*;
use std::fs;

/// Helper function to compare SVGs and provide detailed diff on failure
fn compare_svgs(ts_svg: &str, rust_svg: &str, test_name: &str) {
    if ts_svg == rust_svg {
        println!("✓ {} - SVG outputs are BYTE-FOR-BYTE IDENTICAL!", test_name);
        return;
    }

    // Find first difference
    let ts_lines: Vec<&str> = ts_svg.lines().collect();
    let rust_lines: Vec<&str> = rust_svg.lines().collect();

    eprintln!("\n=== DIFFERENCE FOUND in {} ===", test_name);
    eprintln!("TypeScript lines: {}", ts_lines.len());
    eprintln!("Rust lines: {}", rust_lines.len());

    // Find first differing line
    for (i, (ts_line, rust_line)) in ts_lines.iter().zip(rust_lines.iter()).enumerate() {
        if ts_line != rust_line {
            eprintln!("\nFirst difference at line {}:", i + 1);
            eprintln!("TS:   {}", ts_line);
            eprintln!("Rust: {}", rust_line);

            // Show character-by-character difference
            for (j, (ts_char, rust_char)) in ts_line.chars().zip(rust_line.chars()).enumerate() {
                if ts_char != rust_char {
                    eprintln!("  First char diff at column {}: '{}' vs '{}'", j, ts_char, rust_char);
                    break;
                }
            }

            // Show context (previous and next lines)
            if i > 0 {
                eprintln!("\nPrevious line ({}): {}", i, ts_lines[i - 1]);
            }
            if i + 1 < ts_lines.len().min(rust_lines.len()) {
                eprintln!("Next line ({}): {}", i + 2, ts_lines[i + 1]);
            }

            break;
        }
    }

    if ts_lines.len() != rust_lines.len() {
        eprintln!("\n⚠️  Line count mismatch!");
        if ts_lines.len() > rust_lines.len() {
            eprintln!("Extra TS lines starting at {}:", rust_lines.len() + 1);
            for line in &ts_lines[rust_lines.len()..ts_lines.len().min(rust_lines.len() + 5)] {
                eprintln!("  {}", line);
            }
        } else {
            eprintln!("Extra Rust lines starting at {}:", ts_lines.len() + 1);
            for line in &rust_lines[ts_lines.len()..rust_lines.len().min(ts_lines.len() + 5)] {
                eprintln!("  {}", line);
            }
        }
    }

    panic!("SVG outputs are NOT byte-for-byte identical for {}!", test_name);
}

/// Helper to run a test for a specific mega-complex function
fn test_mega_complex_function(func_idx: usize) {
    let test_name = format!("mega-complex-func{}-pass0", func_idx);
    let ts_svg_path = format!("tests/fixtures/ts-{}.svg", test_name);
    let json_path = "tests/fixtures/mega-complex.json";

    // Read the TypeScript-generated SVG
    let ts_svg = fs::read_to_string(&ts_svg_path)
        .unwrap_or_else(|_| panic!("TypeScript SVG file not found: {} - run ./generate_all_mega_complex.sh first", ts_svg_path));

    // Read and parse the JSON (only once, cached by OS)
    let json_str = fs::read_to_string(json_path).expect("Failed to read mega-complex.json");
    let ion_json: IonJSON = serde_json::from_str(&json_str).expect("Failed to parse JSON");

    // Generate Rust SVG with same input
    let func = &ion_json.functions[func_idx];
    let pass = func.passes[0].clone();

    let mut graph = Graph::new(Vec2::new(5000.0, 5000.0), pass);
    let rust_svg = graph.render_svg();

    // Write Rust output for debugging
    let rust_output_path = format!("tests/fixtures/rust-{}.svg", test_name);
    fs::write(&rust_output_path, &rust_svg).ok();

    // Compare byte-for-byte
    compare_svgs(&ts_svg, &rust_svg, &test_name);
}

// Generate a test for each mega-complex function
#[test]
fn test_mega_complex_func0() { test_mega_complex_function(0); }

#[test]
fn test_mega_complex_func1() { test_mega_complex_function(1); }

#[test]
fn test_mega_complex_func2() { test_mega_complex_function(2); }

#[test]
fn test_mega_complex_func3() { test_mega_complex_function(3); }

#[test]
fn test_mega_complex_func4() { test_mega_complex_function(4); }

#[test]
fn test_mega_complex_func5() { test_mega_complex_function(5); }

#[test]
fn test_mega_complex_func6() { test_mega_complex_function(6); }

#[test]
fn test_mega_complex_func7() { test_mega_complex_function(7); }

#[test]
fn test_mega_complex_func8() { test_mega_complex_function(8); }

#[test]
fn test_mega_complex_func9() { test_mega_complex_function(9); }

#[test]
fn test_mega_complex_func10() { test_mega_complex_function(10); }

#[test]
fn test_mega_complex_func11() { test_mega_complex_function(11); }

#[test]
fn test_mega_complex_func12() { test_mega_complex_function(12); }

#[test]
fn test_mega_complex_func13() { test_mega_complex_function(13); }

#[test]
fn test_mega_complex_func14() { test_mega_complex_function(14); }
