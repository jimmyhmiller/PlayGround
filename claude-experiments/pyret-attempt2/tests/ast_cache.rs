use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

#[derive(Serialize, Deserialize)]
struct CacheIndex {
    entries: HashMap<String, CacheEntry>,
}

#[derive(Serialize, Deserialize)]
struct CacheEntry {
    hash: String,
    size: usize,
}

/// Compute SHA-256 hash of input code
fn compute_hash(code: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(code.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Normalize JSON for comparison (recursive dictionary sorting and srcloc normalization)
fn normalize_json(value: Value) -> Value {
    match value {
        Value::Object(map) => {
            let mut sorted: Vec<_> = map.into_iter().collect();
            sorted.sort_by(|a, b| a.0.cmp(&b.0));
            Value::Object(
                sorted
                    .into_iter()
                    .map(|(k, v)| (k, normalize_json(v)))
                    .collect(),
            )
        }
        Value::Array(arr) => Value::Array(arr.into_iter().map(normalize_json).collect()),
        Value::String(s) => {
            // Normalize srcloc strings to ignore filename differences
            // Format: srcloc("filename", line, col, char, line, col, char)
            if s.starts_with("srcloc(\"") {
                // Find the closing quote of the filename
                if let Some(first_quote) = s.find('"') {
                    if let Some(second_quote) = s[first_quote + 1..].find('"') {
                        let second_quote_pos = first_quote + 1 + second_quote;
                        // Replace filename with "file.arr"
                        let rest = &s[second_quote_pos + 1..];
                        return Value::String(format!("srcloc(\"file.arr\"{}", rest));
                    }
                }
            }
            Value::String(s)
        }
        other => other,
    }
}

/// Load or generate cached Pyret AST for given code
/// Uses lazy caching: if not cached, runs Pyret parser and caches the result
pub fn load_or_generate_cached_ast(code: &str) -> Result<Value, String> {
    use std::process::Command;

    let hash = compute_hash(code);
    let cache_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("pyret-cache")
        .join("asts");

    // Ensure cache directory exists
    fs::create_dir_all(&cache_dir)
        .map_err(|e| format!("Failed to create cache directory: {}", e))?;

    let cache_file = cache_dir.join(format!("{}.json", hash));

    // If cached, load it
    if cache_file.exists() {
        let json_str = fs::read_to_string(&cache_file)
            .map_err(|e| format!("Failed to read cache file: {}", e))?;

        let value: Value = serde_json::from_str(&json_str)
            .map_err(|e| format!("Failed to parse cached JSON: {}", e))?;

        return Ok(value);
    }

    // Otherwise, generate it by running Pyret parser
    let temp_file = format!("/tmp/pyret_lazy_cache_input_{}.arr", hash);
    let temp_output = format!("/tmp/pyret_lazy_cache_output_{}.json", hash);

    fs::write(&temp_file, code)
        .map_err(|e| format!("Failed to write temp file: {}", e))?;

    let output = Command::new("node")
        .current_dir("/Users/jimmyhmiller/Documents/Code/open-source/pyret-lang")
        .arg("ast-to-json.jarr")
        .arg(&temp_file)
        .arg(&temp_output)
        .output()
        .map_err(|e| format!("Failed to run Pyret parser: {}", e))?;

    if !output.status.success() {
        return Err(format!(
            "Pyret parser failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    // Read the generated JSON
    let json_str = fs::read_to_string(&temp_output)
        .map_err(|e| format!("Failed to read Pyret output: {}", e))?;

    let value: Value = serde_json::from_str(&json_str)
        .map_err(|e| format!("Failed to parse Pyret JSON: {}", e))?;

    // Cache it for future use
    fs::write(&cache_file, &json_str)
        .map_err(|e| format!("Failed to write cache file: {}", e))?;

    // Clean up temp files
    let _ = fs::remove_file(temp_file);
    let _ = fs::remove_file(temp_output);

    Ok(value)
}

/// Compare Rust AST with cached Pyret AST
/// For now, this uses the to_pyret_json binary as a subprocess
/// TODO: Refactor to_pyret_json.rs into a library module
pub fn compare_with_cached_pyret(code: &str) -> Result<bool, String> {
    use std::process::Command;

    // Compute hash for unique temp file
    let hash = compute_hash(code);

    // Load or generate cached Pyret AST
    let pyret_ast = load_or_generate_cached_ast(code)?;

    // Write code to unique temp file (use hash to avoid collisions in parallel tests)
    let temp_file = format!("/tmp/rust_parser_test_{}.arr", hash);
    fs::write(&temp_file, code)
        .map_err(|e| format!("Failed to write temp file: {}", e))?;

    // Parse with Rust parser using to_pyret_json binary
    let output = Command::new(env!("CARGO"))
        .args(&["run", "--bin", "to_pyret_json", &temp_file])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .map_err(|e| format!("Failed to run to_pyret_json: {}", e))?;

    if !output.status.success() {
        return Err(format!(
            "Rust parser failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    let rust_json_str = String::from_utf8_lossy(&output.stdout);
    let rust_ast: Value = serde_json::from_str(&rust_json_str)
        .map_err(|e| format!("Failed to parse Rust JSON: {}", e))?;

    // Normalize both for comparison
    let pyret_normalized = normalize_json(pyret_ast);
    let rust_normalized = normalize_json(rust_ast);

    let matches = pyret_normalized == rust_normalized;

    // Debug output if they don't match
    if !matches {
        eprintln!("\n=== AST MISMATCH DEBUG ===");
        eprintln!("Code: {}", code);

        // Write normalized ASTs to temp files for comparison
        if let Ok(pyret_str) = serde_json::to_string_pretty(&pyret_normalized) {
            let _ = fs::write("/tmp/rust_test_pyret_normalized.json", pyret_str);
        }
        if let Ok(rust_str) = serde_json::to_string_pretty(&rust_normalized) {
            let _ = fs::write("/tmp/rust_test_rust_normalized.json", rust_str);
        }

        eprintln!("Wrote normalized ASTs to:");
        eprintln!("  /tmp/rust_test_pyret_normalized.json");
        eprintln!("  /tmp/rust_test_rust_normalized.json");
        eprintln!("Run: diff /tmp/rust_test_pyret_normalized.json /tmp/rust_test_rust_normalized.json");
    }

    Ok(matches)
}
