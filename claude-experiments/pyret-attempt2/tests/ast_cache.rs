use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use tempfile::NamedTempFile;

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
    // Use tempfile for automatic cleanup
    let mut temp_input = NamedTempFile::with_suffix(".arr")
        .map_err(|e| format!("Failed to create temp input file: {}", e))?;

    let temp_output = NamedTempFile::with_suffix(".json")
        .map_err(|e| format!("Failed to create temp output file: {}", e))?;

    temp_input
        .write_all(code.as_bytes())
        .map_err(|e| format!("Failed to write temp input file: {}", e))?;
    temp_input
        .flush()
        .map_err(|e| format!("Failed to flush temp input file: {}", e))?;

    let input_path = temp_input.path();
    let output_path = temp_output.path();

    let pyret_json_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("pyret-json");

    let output = Command::new("node")
        .current_dir(&pyret_json_dir)
        .arg("ast-to-json.jarr")
        .arg(input_path)
        .arg(output_path)
        .output()
        .map_err(|e| format!("Failed to run Pyret parser: {}", e))?;

    if !output.status.success() {
        return Err(format!(
            "Pyret parser failed: {}",
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    // Read the generated JSON
    let json_str = fs::read_to_string(output_path)
        .map_err(|e| format!("Failed to read Pyret output: {}", e))?;

    let value: Value = serde_json::from_str(&json_str)
        .map_err(|e| format!("Failed to parse Pyret JSON: {}", e))?;

    // Cache it for future use
    fs::write(&cache_file, &json_str).map_err(|e| format!("Failed to write cache file: {}", e))?;

    // NamedTempFile automatically cleans up when dropped

    Ok(value)
}

/// Compare Rust AST with cached Pyret AST
/// Parses code directly using the library, no subprocess needed
pub fn compare_with_cached_pyret(code: &str) -> Result<bool, String> {
    use pyret_attempt2::pyret_json::program_to_pyret_json;
    use pyret_attempt2::{FileRegistry, Parser, Tokenizer};

    // Load or generate cached Pyret AST
    let pyret_ast = load_or_generate_cached_ast(code)?;

    // Parse with Rust parser directly (no subprocess!)
    let mut registry = FileRegistry::new();
    let file_id = registry.register("test.arr".to_string());

    let mut tokenizer = Tokenizer::new(code, file_id);
    let tokens = tokenizer.tokenize();
    let mut parser = Parser::new(tokens, file_id);

    let program = parser
        .parse_program()
        .map_err(|e| format!("Rust parser failed: {:?}", e))?;

    let rust_ast = program_to_pyret_json(&program, &registry);

    // Normalize both for comparison
    let pyret_normalized = normalize_json(pyret_ast);
    let rust_normalized = normalize_json(rust_ast);

    let matches = pyret_normalized == rust_normalized;

    // Debug output if they don't match
    if !matches {
        eprintln!("\n=== AST MISMATCH DEBUG ===");
        eprintln!("Code: {}", code);

        // Write normalized ASTs to project debug directory for comparison
        let debug_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("test-debug");

        if let Err(e) = fs::create_dir_all(&debug_dir) {
            eprintln!("Warning: Failed to create debug directory: {}", e);
        } else {
            let pyret_file = debug_dir.join("pyret_normalized.json");
            let rust_file = debug_dir.join("rust_normalized.json");

            if let Ok(pyret_str) = serde_json::to_string_pretty(&pyret_normalized) {
                let _ = fs::write(&pyret_file, pyret_str);
            }
            if let Ok(rust_str) = serde_json::to_string_pretty(&rust_normalized) {
                let _ = fs::write(&rust_file, rust_str);
            }

            eprintln!("Wrote normalized ASTs to:");
            eprintln!("  {}", pyret_file.display());
            eprintln!("  {}", rust_file.display());
            eprintln!("Run: diff {} {}", pyret_file.display(), rust_file.display());
        }
    }

    Ok(matches)
}
