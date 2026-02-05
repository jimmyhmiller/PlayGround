use anyhow::{Context, Result};
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::fs;
use std::path::PathBuf;
use std::time::SystemTime;

// Thread-local overrides for test isolation
thread_local! {
    static SESSION_DIR_OVERRIDE: RefCell<Option<PathBuf>> = const { RefCell::new(None) };
    static SOCKET_DIR_OVERRIDE: RefCell<Option<PathBuf>> = const { RefCell::new(None) };
}

/// Set custom session directory (for testing)
pub fn set_session_dir(path: Option<PathBuf>) {
    SESSION_DIR_OVERRIDE.with(|cell| {
        *cell.borrow_mut() = path;
    });
}

/// Set custom socket directory (for testing)
pub fn set_socket_dir(path: Option<PathBuf>) {
    SOCKET_DIR_OVERRIDE.with(|cell| {
        *cell.borrow_mut() = path;
    });
}

/// Get the current session directory override (for passing to daemon)
pub fn get_session_dir_override() -> Option<PathBuf> {
    SESSION_DIR_OVERRIDE.with(|cell| cell.borrow().clone())
}

/// Get the current socket directory override (for passing to daemon)
pub fn get_socket_dir_override() -> Option<PathBuf> {
    SOCKET_DIR_OVERRIDE.with(|cell| cell.borrow().clone())
}

const ADJECTIVES: &[&str] = &[
    "fuzzy", "quick", "lazy", "happy", "sleepy", "brave", "calm", "eager",
    "gentle", "kind", "lively", "merry", "nice", "proud", "silly", "witty",
    "bold", "cool", "dapper", "fancy", "jolly", "keen", "lucky", "noble",
];

const NOUNS: &[&str] = &[
    "penguin", "dolphin", "falcon", "tiger", "panda", "koala", "otter", "fox",
    "owl", "bear", "wolf", "eagle", "shark", "whale", "raven", "lynx",
    "badger", "gecko", "lemur", "moose", "orca", "quail", "sloth", "zebra",
];

/// Session metadata stored on disk
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionInfo {
    pub name: String,
    pub command: Vec<String>,
    pub pid: u32,
    pub created_at: u64,
    pub socket_path: String,
}

/// Get the directory for session metadata
pub fn sessions_dir() -> Result<PathBuf> {
    // Check for thread-local override first
    if let Some(override_path) = SESSION_DIR_OVERRIDE.with(|cell| cell.borrow().clone()) {
        fs::create_dir_all(&override_path)?;
        return Ok(override_path);
    }

    // Check for environment variable override (for subprocess tests)
    if let Ok(env_path) = std::env::var("KEEP_RUNNING_SESSION_DIR") {
        let path = PathBuf::from(env_path);
        fs::create_dir_all(&path)?;
        return Ok(path);
    }

    let config_dir = dirs::config_dir()
        .context("Could not determine config directory")?
        .join("keep-running")
        .join("sessions");
    fs::create_dir_all(&config_dir)?;
    Ok(config_dir)
}

/// Get the directory for session sockets
pub fn sockets_dir() -> Result<PathBuf> {
    // Check for thread-local override first
    if let Some(override_path) = SOCKET_DIR_OVERRIDE.with(|cell| cell.borrow().clone()) {
        fs::create_dir_all(&override_path)?;
        return Ok(override_path);
    }

    // Check for environment variable override (for subprocess tests)
    if let Ok(env_path) = std::env::var("KEEP_RUNNING_SOCKET_DIR") {
        let path = PathBuf::from(env_path);
        fs::create_dir_all(&path)?;
        return Ok(path);
    }

    let runtime_dir = std::env::var("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            let uid = unsafe { libc::getuid() };
            PathBuf::from(format!("/tmp/keep-running-{}", uid))
        });
    let socket_dir = runtime_dir.join("keep-running");
    fs::create_dir_all(&socket_dir)?;
    Ok(socket_dir)
}

/// Generate a human-readable session name
pub fn generate_name() -> String {
    let mut rng = rand::thread_rng();
    let adj = ADJECTIVES.choose(&mut rng).unwrap();
    let noun = NOUNS.choose(&mut rng).unwrap();
    format!("{}-{}", adj, noun)
}

/// Generate a unique session name (avoid collisions)
pub fn generate_unique_name() -> Result<String> {
    let existing = list_sessions()?;
    let existing_names: std::collections::HashSet<_> =
        existing.iter().map(|s| s.name.as_str()).collect();

    for _ in 0..100 {
        let name = generate_name();
        if !existing_names.contains(name.as_str()) {
            return Ok(name);
        }
    }

    // Fallback: add random suffix
    let name = format!("{}-{}", generate_name(), rand::random::<u16>());
    Ok(name)
}

/// Save session info to disk
pub fn save_session(info: &SessionInfo) -> Result<()> {
    let path = sessions_dir()?.join(format!("{}.json", info.name));
    let json = serde_json::to_string_pretty(info)?;
    fs::write(&path, json)?;
    Ok(())
}

/// Load session info from disk
pub fn load_session(name: &str) -> Result<Option<SessionInfo>> {
    let path = sessions_dir()?.join(format!("{}.json", name));
    if !path.exists() {
        return Ok(None);
    }
    let json = fs::read_to_string(&path)?;
    let info: SessionInfo = serde_json::from_str(&json)?;
    Ok(Some(info))
}

/// Remove session info from disk
pub fn remove_session(name: &str) -> Result<()> {
    let path = sessions_dir()?.join(format!("{}.json", name));
    if path.exists() {
        fs::remove_file(&path)?;
    }

    // Also try to remove socket
    if let Ok(sockets) = sockets_dir() {
        let socket_path = sockets.join(format!("{}.sock", name));
        let _ = fs::remove_file(&socket_path);
    }

    Ok(())
}

/// List all sessions (cleaning up dead ones)
pub fn list_sessions() -> Result<Vec<SessionInfo>> {
    let dir = sessions_dir()?;
    let mut sessions = Vec::new();
    let mut dead_sessions = Vec::new();

    for entry in fs::read_dir(&dir)? {
        let entry = entry?;
        let path = entry.path();

        if path.extension().map(|e| e == "json").unwrap_or(false) {
            if let Ok(json) = fs::read_to_string(&path) {
                if let Ok(info) = serde_json::from_str::<SessionInfo>(&json) {
                    // Check if daemon is still alive
                    if is_process_alive(info.pid) {
                        sessions.push(info);
                    } else {
                        dead_sessions.push(info.name.clone());
                    }
                }
            }
        }
    }

    // Clean up dead sessions
    for name in dead_sessions {
        let _ = remove_session(&name);
    }

    sessions.sort_by(|a, b| a.created_at.cmp(&b.created_at));
    Ok(sessions)
}

/// Check if a process is still running
fn is_process_alive(pid: u32) -> bool {
    unsafe { libc::kill(pid as i32, 0) == 0 }
}

/// Find a session by name (supports prefix matching)
pub fn find_session(query: &str) -> Result<Option<SessionInfo>> {
    let sessions = list_sessions()?;

    // Exact match first
    if let Some(session) = sessions.iter().find(|s| s.name == query) {
        return Ok(Some(session.clone()));
    }

    // Prefix match
    let matches: Vec<_> = sessions
        .iter()
        .filter(|s| s.name.starts_with(query))
        .collect();

    match matches.len() {
        0 => Ok(None),
        1 => Ok(Some(matches[0].clone())),
        _ => {
            let names: Vec<_> = matches.iter().map(|s| s.name.as_str()).collect();
            anyhow::bail!("Ambiguous session name '{}', matches: {}", query, names.join(", "));
        }
    }
}

/// Get socket path for a session
pub fn socket_path(name: &str) -> Result<PathBuf> {
    Ok(sockets_dir()?.join(format!("{}.sock", name)))
}

/// Get current timestamp as seconds since epoch
pub fn timestamp() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}
