use assert_cmd::Command;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::TempDir;

#[derive(Deserialize)]
struct Scenario {
    name: String,
    steps: Vec<Step>,
}

#[derive(Deserialize)]
struct Step {
    cmd: Vec<String>,
    capture: Option<String>,
    expect_stdout_contains: Option<Vec<String>>,
    expect_json_contains: Option<Vec<Value>>,
}

#[test]
fn data_driven_integration() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    let entries = fs::read_dir(&root).expect("tests/data missing");

    for entry in entries {
        let entry = entry.expect("read_dir entry");
        if entry.path().extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        run_scenario(&entry.path());
    }
}

fn run_scenario(path: &Path) {
    let content = fs::read_to_string(path).expect("read scenario");
    let scenario: Scenario = serde_json::from_str(&content).expect("parse scenario");
    let temp = TempDir::new().expect("temp dir");
    let home = temp.path().join("home");
    fs::create_dir_all(&home).expect("create home");

    let mut vars: HashMap<String, String> = HashMap::new();

    for (idx, step) in scenario.steps.iter().enumerate() {
        let cmd = substitute_vars(&step.cmd, &vars);
        let mut command = Command::new(assert_cmd::cargo::cargo_bin!("scope"));
        command.args(&cmd);
        command.env("HOME", &home);
        command.env("USER", "tester");
        command.current_dir(temp.path());

        let output = command.output().expect("run command");
        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        if !output.status.success() {
            panic!(
                "scenario '{}' step {} failed: {:?}\nstdout:\n{}\nstderr:\n{}",
                scenario.name, idx + 1, cmd, stdout, stderr
            );
        }

        if let Some(name) = &step.capture {
            let value = stdout.trim().to_string();
            vars.insert(name.clone(), value);
        }

        if let Some(expect) = &step.expect_stdout_contains {
            for item in substitute_vars(expect, &vars) {
                assert!(
                    stdout.contains(&item),
                    "scenario '{}' step {} missing stdout '{}'\nstdout:\n{}",
                    scenario.name,
                    idx + 1,
                    item,
                    stdout
                );
            }
        }

        if let Some(expect_json) = &step.expect_json_contains {
            let value: Value = serde_json::from_str(&stdout).unwrap_or_else(|_| json!([]));
            for expected in expect_json {
                assert!(
                    json_contains(&value, expected),
                    "scenario '{}' step {} missing json {:?}\nstdout:\n{}",
                    scenario.name,
                    idx + 1,
                    expected,
                    stdout
                );
            }
        }
    }
}

fn substitute_vars(items: &[String], vars: &HashMap<String, String>) -> Vec<String> {
    items
        .iter()
        .map(|s| {
            let mut out = s.clone();
            for (k, v) in vars {
                out = out.replace(&format!("${{{}}}", k), v);
            }
            out
        })
        .collect()
}

fn json_contains(haystack: &Value, needle: &Value) -> bool {
    match (haystack, needle) {
        (Value::Array(items), Value::Object(_)) => {
            items.iter().any(|i| json_contains(i, needle))
        }
        (Value::Object(hm), Value::Object(nm)) => {
            nm.iter().all(|(k, nv)| hm.get(k).map(|hv| json_contains(hv, nv)).unwrap_or(false))
        }
        _ => haystack == needle,
    }
}
