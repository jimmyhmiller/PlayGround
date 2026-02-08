//! Integration test runner
//!
//! Discovers and runs all scenario files from the scenarios/ directory.

mod harness;
mod parser;

use std::path::PathBuf;
use walkdir::WalkDir;

/// Discover all scenario files in the scenarios directory
fn discover_scenarios() -> Vec<PathBuf> {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let scenarios_dir = PathBuf::from(&manifest_dir).join("scenarios");

    if !scenarios_dir.exists() {
        eprintln!("Warning: scenarios directory not found at {}", scenarios_dir.display());
        return Vec::new();
    }

    let mut paths: Vec<PathBuf> = WalkDir::new(&scenarios_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path().extension().map(|ext| ext == "toml").unwrap_or(false)
        })
        .map(|e| e.path().to_path_buf())
        .collect();

    // Sort for deterministic order
    paths.sort();
    paths
}

/// Get scenario name filter from environment
fn get_scenario_filter() -> Option<String> {
    std::env::var("SCENARIO_FILTER").ok()
}

/// Run all discovered scenarios
#[test]
fn run_all_scenarios() {
    let scenarios = discover_scenarios();
    let filter = get_scenario_filter();

    if scenarios.is_empty() {
        println!("No scenarios found. Make sure scenarios/ directory exists with .toml files.");
        return;
    }

    let mut passed = 0;
    let mut failed = 0;
    let mut skipped = 0;
    let mut failures: Vec<(String, String)> = Vec::new();

    for path in &scenarios {
        // Small delay between tests to ensure cleanup
        std::thread::sleep(std::time::Duration::from_millis(100));

        let scenario_name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown");

        // Check filter
        if let Some(ref f) = filter {
            if !scenario_name.contains(f) {
                skipped += 1;
                continue;
            }
        }

        print!("Running scenario: {} ... ", scenario_name);

        match parser::parse_scenario_file(path) {
            Ok(scenario) => {
                match harness::run_scenario(&scenario) {
                    Ok(()) => {
                        println!("PASSED");
                        passed += 1;
                    }
                    Err(e) => {
                        println!("FAILED");
                        println!("  Error: {}", e);
                        failed += 1;
                        failures.push((scenario_name.to_string(), e));
                    }
                }
            }
            Err(e) => {
                println!("PARSE ERROR");
                println!("  Error: {}", e);
                failed += 1;
                failures.push((scenario_name.to_string(), e));
            }
        }
    }

    println!();
    println!("=== Test Summary ===");
    println!("Passed:  {}", passed);
    println!("Failed:  {}", failed);
    println!("Skipped: {}", skipped);
    println!("Total:   {}", scenarios.len());

    if !failures.is_empty() {
        println!();
        println!("=== Failures ===");
        for (name, error) in &failures {
            println!();
            println!("  {}: {}", name, error);
        }

        panic!("{} scenario(s) failed", failed);
    }
}

/// Individual test for echo_hello scenario
#[test]
fn test_echo_hello() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let path = PathBuf::from(&manifest_dir).join("scenarios/basic/echo_hello.toml");

    if !path.exists() {
        println!("Scenario file not found, skipping: {}", path.display());
        return;
    }

    let scenario = parser::parse_scenario_file(&path).expect("Failed to parse scenario");
    harness::run_scenario(&scenario).expect("Scenario failed");
}

/// Individual test for cat_echo scenario
#[test]
fn test_cat_echo() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let path = PathBuf::from(&manifest_dir).join("scenarios/basic/cat_echo.toml");

    if !path.exists() {
        println!("Scenario file not found, skipping: {}", path.display());
        return;
    }

    let scenario = parser::parse_scenario_file(&path).expect("Failed to parse scenario");
    harness::run_scenario(&scenario).expect("Scenario failed");
}

/// Individual test for simple_reattach scenario
#[test]
fn test_simple_reattach() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let path = PathBuf::from(&manifest_dir).join("scenarios/reattach/simple_reattach.toml");

    if !path.exists() {
        println!("Scenario file not found, skipping: {}", path.display());
        return;
    }

    let scenario = parser::parse_scenario_file(&path).expect("Failed to parse scenario");
    harness::run_scenario(&scenario).expect("Scenario failed");
}

/// Individual test for multi_cycle scenario
#[test]
fn test_multi_cycle() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let path = PathBuf::from(&manifest_dir).join("scenarios/reattach/multi_cycle.toml");

    if !path.exists() {
        println!("Scenario file not found, skipping: {}", path.display());
        return;
    }

    let scenario = parser::parse_scenario_file(&path).expect("Failed to parse scenario");
    harness::run_scenario(&scenario).expect("Scenario failed");
}

/// Test that multi-line paste doesn't lock up
#[test]
fn test_multiline_paste() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let path = PathBuf::from(&manifest_dir).join("scenarios/edge_cases/multiline_paste.toml");

    if !path.exists() {
        println!("Scenario file not found, skipping: {}", path.display());
        return;
    }

    let scenario = parser::parse_scenario_file(&path).expect("Failed to parse scenario");
    harness::run_scenario(&scenario).expect("Scenario failed");
}
