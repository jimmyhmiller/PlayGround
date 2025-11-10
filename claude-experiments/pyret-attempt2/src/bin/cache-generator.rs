use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::process::Command;

#[derive(Serialize, Deserialize)]
struct CacheIndex {
    entries: HashMap<String, CacheEntry>,
}

#[derive(Serialize, Deserialize)]
struct CacheEntry {
    hash: String,
    size: usize,
}

fn compute_hash(content: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(content.as_bytes());
    format!("{:x}", hasher.finalize())
}

fn run_pyret_parser(code: &str) -> Result<String, String> {
    // Write input to temp file
    let input_path = "/tmp/pyret_cache_gen_input.arr";
    fs::write(input_path, code).map_err(|e| format!("Failed to write input: {}", e))?;

    // Run Pyret parser
    let output = Command::new("node")
        .current_dir("/Users/jimmyhmiller/Documents/Code/open-source/pyret-lang")
        .arg("ast-to-json.jarr")
        .arg(input_path)
        .arg("/tmp/pyret_cache_gen_output.json")
        .output()
        .map_err(|e| format!("Failed to run Pyret parser: {}", e))?;

    if !output.status.success() {
        return Err(format!(
            "Pyret parser failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    // Read output
    let json = fs::read_to_string("/tmp/pyret_cache_gen_output.json")
        .map_err(|e| format!("Failed to read output: {}", e))?;

    Ok(json)
}

fn main() {
    println!("Pyret AST Cache Generator");
    println!("=========================\n");

    // Read comparison_tests.rs and extract test inputs
    let test_file = fs::read_to_string("tests/comparison_tests.rs")
        .expect("Failed to read comparison_tests.rs");

    let mut test_inputs: Vec<(String, String)> = Vec::new();
    let mut current_test_name = String::new();

    // Extract inline test cases
    for line in test_file.lines() {
        // Track test function names
        if line.trim().starts_with("fn test_") {
            if let Some(name_end) = line.find('(') {
                if let Some(name_start) = line.find("fn ") {
                    current_test_name = line[name_start + 3..name_end].to_string();
                }
            }
        }

        // Extract assert_matches_pyret calls with regular string literals
        if let Some(start) = line.find("assert_matches_pyret(\"") {
            let code_start = start + "assert_matches_pyret(\"".len();
            // Find the matching closing quote, handling escaped quotes
            let mut end_pos = code_start;
            let mut found_end = false;
            let chars: Vec<char> = line.chars().collect();

            let mut i = code_start;
            while i < chars.len() {
                if chars[i] == '\\' && i + 1 < chars.len() {
                    // Skip escaped character
                    i += 2;
                } else if chars[i] == '"' && i + 1 < chars.len() && chars[i + 1] == ')' {
                    // Found the end
                    end_pos = i;
                    found_end = true;
                    break;
                } else {
                    i += 1;
                }
            }

            if found_end {
                let code: String = chars[code_start..end_pos].iter().collect();
                // Unescape the string for parsing
                let code = code.replace("\\\"", "\"").replace("\\\\", "\\");
                let test_name = if current_test_name.is_empty() {
                    format!("inline_{}", test_inputs.len())
                } else {
                    format!("{}_{}", current_test_name, test_inputs.len())
                };
                test_inputs.push((test_name, code));
            }
        }

        // Extract file-based tests with include_str!
        if let Some(start) = line.find("include_str!(\"") {
            let path_start = start + "include_str!(\"".len();
            if let Some(end) = line[path_start..].find("\")") {
                let relative_path = &line[path_start..path_start + end];
                let full_path = format!("tests/{}", relative_path);

                if let Ok(code) = fs::read_to_string(&full_path) {
                    let test_name = if current_test_name.is_empty() {
                        format!("file_{}", test_inputs.len())
                    } else {
                        current_test_name.clone()
                    };
                    test_inputs.push((test_name, code));
                }
            }
        }
    }

    println!("Found {} test inputs\n", test_inputs.len());

    let mut cache_index = CacheIndex {
        entries: HashMap::new(),
    };

    let mut success_count = 0;
    let mut error_count = 0;

    for (i, (test_name, code)) in test_inputs.iter().enumerate() {
        let hash = compute_hash(code);
        let cache_path = format!("tests/pyret-cache/asts/{}.json", hash);

        // Skip if already cached
        if fs::metadata(&cache_path).is_ok() {
            println!("[{}/{}] ✓ {} (cached)", i + 1, test_inputs.len(), test_name);
            cache_index.entries.insert(
                test_name.clone(),
                CacheEntry {
                    hash: hash.clone(),
                    size: code.len(),
                },
            );
            success_count += 1;
            continue;
        }

        print!("[{}/{}] Generating {} ... ", i + 1, test_inputs.len(), test_name);
        std::io::stdout().flush().unwrap();

        match run_pyret_parser(code) {
            Ok(json) => {
                fs::write(&cache_path, &json).expect("Failed to write cache file");
                cache_index.entries.insert(
                    test_name.clone(),
                    CacheEntry {
                        hash: hash.clone(),
                        size: code.len(),
                    },
                );
                println!("✓ ({} bytes)", json.len());
                success_count += 1;
            }
            Err(e) => {
                println!("✗ Error: {}", e);
                error_count += 1;
            }
        }
    }

    // Write cache index
    let index_json = serde_json::to_string_pretty(&cache_index).unwrap();
    fs::write("tests/pyret-cache/cache-index.json", index_json)
        .expect("Failed to write cache index");

    println!("\n=========================");
    println!("Cache generation complete!");
    println!("Success: {}", success_count);
    println!("Errors: {}", error_count);
    println!("Cache location: tests/pyret-cache/");
}
