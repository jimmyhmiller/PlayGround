//! TOML scenario file parser
//!
//! Parses test scenario definitions from TOML files into Rust structs.

use serde::Deserialize;
use std::fs;
use std::path::Path;

/// A complete test scenario
#[derive(Debug, Clone, Deserialize)]
pub struct Scenario {
    pub name: String,
    #[allow(dead_code)]
    pub description: String,
    #[serde(default = "default_timeout")]
    pub timeout_secs: u64,
    pub command: Command,
    #[serde(default)]
    pub actions: Vec<Action>,
    pub expected_final_state: Option<ExpectedFinalState>,
}

fn default_timeout() -> u64 {
    30
}

/// Command to run in the session
#[derive(Debug, Clone, Deserialize)]
pub struct Command {
    pub program: String,
    #[serde(default)]
    pub args: Vec<String>,
}

/// Action types that can be performed during a test
#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Action {
    /// Attach to the session
    Attach {
        #[serde(default)]
        expect_replay_contains: Option<String>,
    },

    /// Send text input
    SendInput {
        text: String,
    },

    /// Send raw bytes
    SendBytes {
        bytes: Vec<u8>,
    },

    /// Wait for specific output pattern
    WaitForOutput {
        #[serde(default)]
        contains: Option<String>,
        #[serde(default)]
        regex: Option<String>,
        #[serde(default = "default_action_timeout")]
        timeout_secs: u64,
    },

    /// Wait for child process to exit
    WaitForExit {
        #[serde(default)]
        expected_code: Option<i32>,
        #[serde(default = "default_action_timeout")]
        timeout_secs: u64,
    },

    /// Detach from the session (send Detach message)
    Detach,

    /// Disconnect without sending Detach message
    DisconnectRaw,

    /// Sleep for a duration
    Sleep {
        duration_ms: u64,
    },

    /// Resize the terminal
    Resize {
        cols: u16,
        rows: u16,
    },

    /// Assert that session file exists
    AssertSessionExists {
        #[serde(default)]
        name: Option<String>,
    },

    /// Assert that session file is gone
    AssertSessionGone {
        #[serde(default)]
        name: Option<String>,
    },
}

fn default_action_timeout() -> u64 {
    5
}

/// Expected state at the end of the test
#[derive(Debug, Clone, Deserialize)]
pub struct ExpectedFinalState {
    #[serde(default)]
    pub daemon_alive: Option<bool>,
    #[serde(default)]
    pub session_file_exists: Option<bool>,
}

/// Parse a scenario from a TOML file
pub fn parse_scenario_file(path: &Path) -> Result<Scenario, String> {
    let content = fs::read_to_string(path)
        .map_err(|e| format!("Failed to read {}: {}", path.display(), e))?;

    let scenario: Scenario = toml::from_str(&content)
        .map_err(|e| format!("Failed to parse {}: {}", path.display(), e))?;

    validate_scenario(&scenario)?;

    Ok(scenario)
}

/// Validate scenario structure
fn validate_scenario(scenario: &Scenario) -> Result<(), String> {
    if scenario.name.is_empty() {
        return Err("Scenario name cannot be empty".to_string());
    }

    if scenario.command.program.is_empty() {
        return Err("Command program cannot be empty".to_string());
    }

    // Validate actions
    for (i, action) in scenario.actions.iter().enumerate() {
        match action {
            Action::WaitForOutput {
                contains, regex, ..
            } => {
                if contains.is_none() && regex.is_none() {
                    return Err(format!(
                        "Action {} (wait_for_output): must specify 'contains' or 'regex'",
                        i
                    ));
                }
            }
            Action::SendInput { text } => {
                if text.is_empty() {
                    return Err(format!(
                        "Action {} (send_input): text cannot be empty",
                        i
                    ));
                }
            }
            _ => {}
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_basic_scenario() {
        let toml = r#"
            name = "test"
            description = "A test scenario"
            timeout_secs = 10

            [command]
            program = "echo"
            args = ["hello"]

            [[actions]]
            type = "attach"

            [[actions]]
            type = "wait_for_output"
            contains = "hello"
            timeout_secs = 5

            [expected_final_state]
            daemon_alive = false
        "#;

        let scenario: Scenario = toml::from_str(toml).unwrap();
        assert_eq!(scenario.name, "test");
        assert_eq!(scenario.command.program, "echo");
        assert_eq!(scenario.actions.len(), 2);
    }

    #[test]
    fn test_parse_send_bytes() {
        let toml = r#"
            name = "test"
            description = "Test send bytes"

            [command]
            program = "cat"

            [[actions]]
            type = "send_bytes"
            bytes = [4]
        "#;

        let scenario: Scenario = toml::from_str(toml).unwrap();
        match &scenario.actions[0] {
            Action::SendBytes { bytes } => {
                assert_eq!(bytes, &[4u8]);
            }
            _ => panic!("Expected SendBytes action"),
        }
    }
}
