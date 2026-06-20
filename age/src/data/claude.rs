//! A [`WorldSource`] backed by Claude Code's local session logs.
//!
//! Layout it reads (override the root with `CLAUDE_PROJECTS_DIR`):
//!
//! ```text
//! ~/.claude/projects/
//!   -Users-me-Documents-Code-foo/        <- one directory per project (a city)
//!     a088fdfe-....jsonl                 <- one JSONL file per session (a building)
//!     8dfeb5b6-....jsonl
//! ```
//!
//! Each JSONL line is one event (`user`, `assistant`, `attachment`, `ai-title`,
//! `file-history-snapshot`, ...). We accumulate cheap per-session stats and use the
//! file's mtime as "last active" so we can tell which sessions are live right now.
//!
//! Parsing is cached by `(mtime, size)`, so polling 76 projects with multi-MB logs
//! only re-reads the handful of files that actually changed since the last poll.

use super::{CityInfo, SessionInfo, WorldSnapshot, WorldSource};
use crate::util::{now_unix, parse_rfc3339};
use std::collections::HashMap;
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

pub struct ClaudeProjectsSource {
    root: PathBuf,
    /// path -> (mtime_ns, size, parsed). Lets us skip unchanged files.
    cache: HashMap<PathBuf, CacheEntry>,
}

struct CacheEntry {
    mtime_ns: u128,
    size: u64,
    info: SessionInfo,
}

impl ClaudeProjectsSource {
    /// Build a source rooted at `~/.claude/projects` (or `$CLAUDE_PROJECTS_DIR`).
    pub fn new() -> ClaudeProjectsSource {
        let root = std::env::var_os("CLAUDE_PROJECTS_DIR")
            .map(PathBuf::from)
            .or_else(|| dirs::home_dir().map(|h| h.join(".claude").join("projects")))
            .unwrap_or_else(|| PathBuf::from(".claude/projects"));
        ClaudeProjectsSource { root, cache: HashMap::new() }
    }
}

impl WorldSource for ClaudeProjectsSource {
    fn name(&self) -> &str {
        "claude-projects"
    }

    fn poll(&mut self) -> WorldSnapshot {
        let mut cities = Vec::new();
        let entries = match fs::read_dir(&self.root) {
            Ok(e) => e,
            Err(_) => {
                // No projects dir — return empty rather than crash; the HUD will say so.
                return WorldSnapshot { cities, captured_at: now_unix() };
            }
        };

        // Keep the cache from growing without bound: forget files we don't see.
        let mut seen: Vec<PathBuf> = Vec::new();

        for proj in entries.flatten() {
            let proj_path = proj.path();
            if !proj_path.is_dir() {
                continue;
            }
            let mut sessions = Vec::new();
            let mut city_name: Option<String> = None;
            let mut city_path: Option<String> = None;

            let session_files = match fs::read_dir(&proj_path) {
                Ok(e) => e,
                Err(_) => continue,
            };
            for sess in session_files.flatten() {
                let p = sess.path();
                if p.extension().and_then(|e| e.to_str()) != Some("jsonl") {
                    continue;
                }
                seen.push(p.clone());
                let info = self.session_info(&p);
                if let Some(info) = info {
                    // The first session that knows the project's cwd names the city.
                    if city_path.is_none() {
                        if let Some((name, path)) = read_project_identity(&p) {
                            city_name = Some(name);
                            city_path = Some(path);
                        }
                    }
                    sessions.push(info);
                }
            }

            if sessions.is_empty() {
                continue;
            }
            // Newest session first.
            sessions.sort_by(|a, b| {
                b.last_active
                    .unwrap_or(0.0)
                    .partial_cmp(&a.last_active.unwrap_or(0.0))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            let id = proj_path
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("project")
                .to_string();
            let name = city_name.unwrap_or_else(|| decode_dir_name(&id));
            cities.push(CityInfo { id, name, path: city_path, sessions });
        }

        // Evict stale cache entries.
        let seen: std::collections::HashSet<_> = seen.into_iter().collect();
        self.cache.retain(|k, _| seen.contains(k));

        // Biggest/busiest cities first for stable, pleasant layout.
        cities.sort_by(|a, b| b.total_messages().cmp(&a.total_messages()));
        WorldSnapshot { cities, captured_at: now_unix() }
    }
}

impl ClaudeProjectsSource {
    /// Parse one session file into [`SessionInfo`], using the `(mtime, size)` cache.
    fn session_info(&mut self, path: &Path) -> Option<SessionInfo> {
        let meta = fs::metadata(path).ok()?;
        let size = meta.len();
        let mtime_ns = meta
            .modified()
            .ok()
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let mtime_secs = mtime_ns as f64 / 1e9;

        if let Some(hit) = self.cache.get(path) {
            if hit.mtime_ns == mtime_ns && hit.size == size {
                return Some(hit.info.clone());
            }
        }

        let info = parse_session(path, mtime_secs)?;
        self.cache.insert(
            path.to_path_buf(),
            CacheEntry { mtime_ns, size, info: info.clone() },
        );
        Some(info)
    }
}

/// Parse a single session JSONL file into accumulated stats.
fn parse_session(path: &Path, mtime_secs: f64) -> Option<SessionInfo> {
    let file = fs::File::open(path).ok()?;
    let reader = BufReader::new(file);

    let id = path.file_stem().and_then(|s| s.to_str()).unwrap_or("session").to_string();
    let mut info = SessionInfo {
        id,
        title: None,
        model: None,
        user_messages: 0,
        assistant_messages: 0,
        tool_uses: 0,
        first_active: None,
        last_active: Some(mtime_secs),
        git_branch: None,
    };

    for line in reader.lines().map_while(Result::ok) {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let v: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };
        let ty = v.get("type").and_then(|t| t.as_str()).unwrap_or("");

        if let Some(ts) = v.get("timestamp").and_then(|t| t.as_str()) {
            if let Some(secs) = parse_rfc3339(ts) {
                info.first_active = Some(info.first_active.map_or(secs, |f: f64| f.min(secs)));
            }
        }
        if info.git_branch.is_none() {
            if let Some(b) = v.get("gitBranch").and_then(|b| b.as_str()) {
                if !b.is_empty() {
                    info.git_branch = Some(b.to_string());
                }
            }
        }

        match ty {
            "ai-title" => {
                if let Some(t) = v.get("aiTitle").and_then(|t| t.as_str()) {
                    info.title = Some(t.to_string());
                }
            }
            "user" => {
                // Skip tool-result echoes: count only real user turns.
                if !is_tool_result(&v) {
                    info.user_messages += 1;
                }
            }
            "assistant" => {
                info.assistant_messages += 1;
                if let Some(m) = v.get("message") {
                    if let Some(model) = m.get("model").and_then(|x| x.as_str()) {
                        if !model.is_empty() {
                            info.model = Some(model.to_string());
                        }
                    }
                    if let Some(content) = m.get("content").and_then(|c| c.as_array()) {
                        for block in content {
                            if block.get("type").and_then(|t| t.as_str()) == Some("tool_use") {
                                info.tool_uses += 1;
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
    Some(info)
}

/// A `user`-typed line whose message content is purely tool results (not a real
/// human turn).
fn is_tool_result(v: &serde_json::Value) -> bool {
    let content = match v.get("message").and_then(|m| m.get("content")) {
        Some(c) => c,
        None => return false,
    };
    match content.as_array() {
        Some(arr) => {
            !arr.is_empty()
                && arr
                    .iter()
                    .all(|b| b.get("type").and_then(|t| t.as_str()) == Some("tool_result"))
        }
        None => false,
    }
}

/// Read a project's friendly name + cwd from the first line that carries `cwd`.
fn read_project_identity(session_path: &Path) -> Option<(String, String)> {
    let file = fs::File::open(session_path).ok()?;
    let reader = BufReader::new(file);
    for line in reader.lines().map_while(Result::ok).take(200) {
        let v: serde_json::Value = match serde_json::from_str(line.trim()) {
            Ok(v) => v,
            Err(_) => continue,
        };
        if let Some(cwd) = v.get("cwd").and_then(|c| c.as_str()) {
            if !cwd.is_empty() {
                let name = Path::new(cwd)
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or(cwd)
                    .to_string();
                return Some((name, cwd.to_string()));
            }
        }
    }
    None
}

/// Fallback name when we can't read a cwd: take the last path-ish segment of the
/// encoded directory name (`-Users-me-Code-foo-bar` -> `bar`). Imperfect because
/// the encoding is lossy (real `-` vs separator), but only a fallback.
fn decode_dir_name(dir: &str) -> String {
    dir.trim_start_matches('-')
        .rsplit('-')
        .next()
        .filter(|s| !s.is_empty())
        .unwrap_or(dir)
        .to_string()
}
