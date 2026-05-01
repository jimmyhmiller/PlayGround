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

/// Test rapid reattach after raw disconnect
#[test]
fn test_rapid_reattach() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let path = PathBuf::from(&manifest_dir).join("scenarios/reattach/rapid_reattach.toml");

    if !path.exists() {
        println!("Scenario file not found, skipping: {}", path.display());
        return;
    }

    let scenario = parser::parse_scenario_file(&path).expect("Failed to parse scenario");
    harness::run_scenario(&scenario).expect("Scenario failed");
}

/// Test reattach with a large replay buffer
#[test]
fn test_reattach_large_buffer() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let path = PathBuf::from(&manifest_dir).join("scenarios/reattach/reattach_with_large_buffer.toml");

    if !path.exists() {
        println!("Scenario file not found, skipping: {}", path.display());
        return;
    }

    let scenario = parser::parse_scenario_file(&path).expect("Failed to parse scenario");
    harness::run_scenario(&scenario).expect("Scenario failed");
}

/// Test high-throughput output during attach
#[test]
fn test_output_during_attach() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let path = PathBuf::from(&manifest_dir).join("scenarios/edge_cases/output_during_attach.toml");

    if !path.exists() {
        println!("Scenario file not found, skipping: {}", path.display());
        return;
    }

    let scenario = parser::parse_scenario_file(&path).expect("Failed to parse scenario");
    harness::run_scenario(&scenario).expect("Scenario failed");
}

/// Test daemon survives multiple raw disconnects
#[test]
fn test_multiple_raw_disconnects() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let path = PathBuf::from(&manifest_dir).join("scenarios/edge_cases/multiple_raw_disconnects.toml");

    if !path.exists() {
        println!("Scenario file not found, skipping: {}", path.display());
        return;
    }

    let scenario = parser::parse_scenario_file(&path).expect("Failed to parse scenario");
    harness::run_scenario(&scenario).expect("Scenario failed");
}

/// Test attaching to session where process already exited
#[test]
fn test_exit_during_attach() {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let path = PathBuf::from(&manifest_dir).join("scenarios/reattach/exit_during_attach.toml");

    if !path.exists() {
        println!("Scenario file not found, skipping: {}", path.display());
        return;
    }

    let scenario = parser::parse_scenario_file(&path).expect("Failed to parse scenario");
    harness::run_scenario(&scenario).expect("Scenario failed");
}

/// Reproduces the "silent zombie client" bug: a client that connects, sends
/// Attach, and then never reads or writes again must not be allowed to keep
/// the daemon alive forever. The daemon should detect the unresponsive peer
/// and clean up after the grace period.
#[test]
fn zombie_client_does_not_block_daemon_cleanup() {
    use keep_running::protocol::{encode_message, ClientMessage};
    use std::io::Write;
    use std::os::unix::net::UnixStream;
    use std::path::PathBuf;
    use std::process::Command;
    use std::time::{Duration, Instant};
    use tempfile::TempDir;

    fn pid_alive(pid: i32) -> bool {
        unsafe { libc::kill(pid, 0) == 0 }
    }

    let temp = TempDir::new().unwrap();
    let session_dir = temp.path().join("sessions");
    let socket_dir = temp.path().join("sockets");
    std::fs::create_dir_all(&session_dir).unwrap();
    std::fs::create_dir_all(&socket_dir).unwrap();

    let session_name = format!("zombie-{}", std::process::id());

    let manifest_dir =
        std::env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string());
    let binary = PathBuf::from(&manifest_dir)
        .join("target")
        .join("debug")
        .join("keep-running");

    // Short grace + short zombie-detection timeout so the test runs in seconds.
    let status = Command::new(&binary)
        .args(["start", "--name", &session_name, "--", "sleep", "0.3"])
        .env("KEEP_RUNNING_SESSION_DIR", &session_dir)
        .env("KEEP_RUNNING_SOCKET_DIR", &socket_dir)
        .env("KEEP_RUNNING_GRACE_SECS", "1")
        .env("KEEP_RUNNING_ZOMBIE_SECS", "1")
        .status()
        .unwrap();
    assert!(status.success(), "start command failed");

    let socket_path = socket_dir.join(format!("{}.sock", session_name));
    let session_file = session_dir.join(format!("{}.json", session_name));

    let deadline = Instant::now() + Duration::from_secs(5);
    while !socket_path.exists() && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(20));
    }
    assert!(socket_path.exists(), "socket never appeared");

    let json: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&session_file).unwrap()).unwrap();
    let daemon_pid = json["pid"].as_i64().unwrap() as i32;
    assert!(pid_alive(daemon_pid), "daemon should be alive after start");

    // Connect as a silent zombie: send Attach, then never read or write
    // anything else and never close the socket.
    let mut stream = UnixStream::connect(&socket_path).unwrap();
    let attach = ClientMessage::Attach { cols: 80, rows: 24 };
    stream.write_all(&encode_message(&attach).unwrap()).unwrap();

    // Timeline:
    //   t=0      child sleep 0.3 still running
    //   t=0.3    child exits
    //   t=0.3+   daemon queues ChildExited, can't drain (we never read)
    //   t=1.3+   zombie-detection timeout elapses → daemon force-disconnects us
    //   t=2.3+   grace period elapses → daemon exits
    // Allow generous slack for CI: 10s.
    let deadline = Instant::now() + Duration::from_secs(10);
    while pid_alive(daemon_pid) && Instant::now() < deadline {
        std::thread::sleep(Duration::from_millis(100));
    }

    // Holding the socket open the whole time — drop only after assertion
    // so we know the daemon really did break out without our help.
    let final_alive = pid_alive(daemon_pid);
    drop(stream);

    if final_alive {
        // Force-cleanup so the test process doesn't leak the daemon.
        unsafe {
            libc::kill(daemon_pid, libc::SIGKILL);
        }
        panic!(
            "daemon (pid {}) was still alive after 10s with a silent zombie client; \
             zombie-detection didn't kick in",
            daemon_pid
        );
    }
}
