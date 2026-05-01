mod cli;
mod client;
mod daemon;
mod error;
mod protocol;
mod session;
mod terminal;

use anyhow::Result;
use cli::Commands;
use std::path::{Path, PathBuf};
use terminal::{status, status_dim};

fn session_not_found_message(query: &str) -> String {
    match session::list_sessions() {
        Ok(sessions) if sessions.is_empty() => {
            format!("session '{}' not found (no sessions running)", query)
        }
        Ok(sessions) => {
            let names: Vec<_> = sessions.iter().map(|s| s.name.as_str()).collect();
            format!(
                "session '{}' not found. Running sessions: {}",
                query,
                names.join(", ")
            )
        }
        Err(_) => format!("session '{}' not found", query),
    }
}

/// Truncate a string to at most `max` characters, appending `...` if truncated.
/// Char-safe (won't panic on multibyte boundaries).
fn truncate_chars(s: &str, max: usize) -> String {
    let count = s.chars().count();
    if count <= max {
        return s.to_string();
    }
    let keep = max.saturating_sub(3);
    let mut out: String = s.chars().take(keep).collect();
    out.push_str("...");
    out
}

/// Format a duration in seconds as "5s", "12m", "3h", "2d".
fn humanize_age(secs: u64) -> String {
    if secs < 60 {
        format!("{}s", secs)
    } else if secs < 3600 {
        format!("{}m", secs / 60)
    } else if secs < 86400 {
        format!("{}h", secs / 3600)
    } else {
        format!("{}d", secs / 86400)
    }
}

/// Resolve a command's binary against PATH so we can fail fast with a clean error
/// instead of letting the daemon's child print "Failed to exec ..." into the PTY.
fn resolve_program(program: &str) -> Result<PathBuf> {
    use std::os::unix::fs::PermissionsExt;

    let path = Path::new(program);
    if program.contains('/') {
        if !path.exists() {
            anyhow::bail!("command not found: {}", program);
        }
        return Ok(path.to_path_buf());
    }

    let path_var = std::env::var_os("PATH")
        .ok_or_else(|| anyhow::anyhow!("PATH is not set"))?;
    for dir in std::env::split_paths(&path_var) {
        let candidate = dir.join(program);
        if let Ok(meta) = candidate.metadata() {
            if meta.is_file() && meta.permissions().mode() & 0o111 != 0 {
                return Ok(candidate);
            }
        }
    }
    anyhow::bail!("command not found: {}", program);
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {}", err);
        for cause in err.chain().skip(1) {
            eprintln!("  caused by: {}", cause);
        }
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let cli = cli::parse();

    match cli.command {
        Some(Commands::Run { name, command }) => cmd_run(name, command),
        Some(Commands::Shell { name }) => cmd_shell(name),
        Some(Commands::Attach { session }) => cmd_attach(&session),
        Some(Commands::List) => cmd_list(),
        Some(Commands::Kill { session }) => cmd_kill(&session),
        Some(Commands::Start { name, command }) => cmd_start(name, command),
        Some(Commands::Completions { shell }) => {
            cli::print_completions(shell);
            Ok(())
        }
        None => {
            // Check if we got a session name as shortcut
            if let Some(session_name) = cli.session {
                cmd_attach(&session_name)
            } else {
                // Default to list
                cmd_list()
            }
        }
    }
}

fn detect_shell() -> String {
    if let Ok(shell) = std::env::var("SHELL") {
        return shell;
    }
    "/bin/sh".to_string()
}

fn cmd_shell(name: Option<String>) -> Result<()> {
    let shell = detect_shell();
    let command = vec![shell];
    cmd_run(name, command)
}

fn check_nested() -> Result<()> {
    if std::env::var("KEEP_RUNNING").is_ok() {
        anyhow::bail!("already inside a keep-running session. Detach first (Ctrl+a d).");
    }
    Ok(())
}

fn cmd_run(name: Option<String>, command: Vec<String>) -> Result<()> {
    check_nested()?;
    if command.is_empty() {
        anyhow::bail!("no command specified");
    }

    // Pre-validate so the user gets a clean error instead of a flash inside the PTY.
    resolve_program(&command[0])?;

    let session_name = name.unwrap_or_else(|| {
        session::generate_unique_name().unwrap_or_else(|_| session::generate_name())
    });

    if session::load_session(&session_name)?.is_some() {
        anyhow::bail!("session '{}' already exists", session_name);
    }

    client::run_and_attach(&session_name, &command)?;

    Ok(())
}

fn cmd_attach(session_query: &str) -> Result<()> {
    let session = match session::find_session(session_query)? {
        Some(s) => s,
        None => anyhow::bail!("{}", session_not_found_message(session_query)),
    };

    status(&format!("attached to '{}' · pid {}", session.name, session.pid));
    status_dim("detach with ctrl-a d  ·  kill with ctrl-a k");

    client::attach(&session)?;

    Ok(())
}

fn cmd_list() -> Result<()> {
    let sessions = session::list_sessions()?;

    if sessions.is_empty() {
        println!("No running sessions.");
        println!();
        println!("Try:");
        println!("  keep-running shell                  start a session running your shell");
        println!("  keep-running run -- <command>       start a session running a command");
        return Ok(());
    }

    let now = session::timestamp();

    println!(
        "{:<20} {:<8} {:<8} {:<8} {}",
        "NAME", "PID", "STATUS", "UPTIME", "COMMAND"
    );

    for s in sessions {
        let age = humanize_age(now.saturating_sub(s.created_at));
        let cmd = s.command.join(" ");
        let cmd_display = truncate_chars(&cmd, 40);
        println!(
            "{:<20} {:<8} {:<8} {:<8} {}",
            truncate_chars(&s.name, 20),
            s.pid,
            "running",
            age,
            cmd_display
        );
    }

    Ok(())
}

fn cmd_kill(session_query: &str) -> Result<()> {
    let session = match session::find_session(session_query)? {
        Some(s) => s,
        None => anyhow::bail!("{}", session_not_found_message(session_query)),
    };

    unsafe {
        libc::kill(session.pid as i32, libc::SIGTERM);
    }

    session::remove_session(&session.name)?;

    status(&format!("killed '{}'", session.name));

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn humanize_age_seconds() {
        assert_eq!(humanize_age(0), "0s");
        assert_eq!(humanize_age(1), "1s");
        assert_eq!(humanize_age(59), "59s");
    }

    #[test]
    fn humanize_age_minutes() {
        assert_eq!(humanize_age(60), "1m");
        assert_eq!(humanize_age(61), "1m");
        assert_eq!(humanize_age(3599), "59m");
    }

    #[test]
    fn humanize_age_hours() {
        assert_eq!(humanize_age(3600), "1h");
        assert_eq!(humanize_age(3661), "1h");
        assert_eq!(humanize_age(86399), "23h");
    }

    #[test]
    fn humanize_age_days() {
        assert_eq!(humanize_age(86400), "1d");
        assert_eq!(humanize_age(172800), "2d");
        assert_eq!(humanize_age(86400 * 30), "30d");
    }

    #[test]
    fn truncate_chars_shorter_than_max_unchanged() {
        assert_eq!(truncate_chars("hi", 10), "hi");
    }

    #[test]
    fn truncate_chars_exactly_max_unchanged() {
        assert_eq!(truncate_chars("hello", 5), "hello");
    }

    #[test]
    fn truncate_chars_longer_appends_ellipsis() {
        assert_eq!(truncate_chars("hello world", 8), "hello...");
        // Result fits within max.
        assert!(truncate_chars("hello world", 8).chars().count() <= 8);
    }

    #[test]
    fn truncate_chars_handles_multibyte_without_panic() {
        // 11 chars but >11 bytes — the original `&s[..n]` slicing approach
        // would panic on a non-char-boundary cut. Char-based truncation
        // must not.
        let s = "héllo wörld";
        let out = truncate_chars(s, 8);
        assert!(out.chars().count() <= 8);
        // The output is still valid UTF-8 (String guarantees this; the real
        // assertion is that we didn't panic above).
        assert!(out.ends_with("..."));
    }

    #[test]
    fn truncate_chars_max_smaller_than_ellipsis() {
        // saturating_sub means max < 3 collapses to just "..." (which itself
        // exceeds max). Documenting current behaviour so it doesn't change
        // silently.
        assert_eq!(truncate_chars("abcdef", 2), "...");
        assert_eq!(truncate_chars("abcdef", 0), "...");
    }

    #[test]
    fn resolve_program_absolute_existing() {
        let f = tempfile::NamedTempFile::new().unwrap();
        let path = f.path().to_path_buf();
        let resolved = resolve_program(path.to_str().unwrap()).unwrap();
        assert_eq!(resolved, path);
    }

    #[test]
    fn resolve_program_absolute_missing() {
        let err = resolve_program("/nonexistent/path/xyzzy-12345").unwrap_err();
        assert!(
            err.to_string().contains("command not found"),
            "got: {err}"
        );
    }

    #[test]
    fn resolve_program_relative_with_slash_missing() {
        // Anything containing a `/` skips PATH lookup and is checked literally.
        let err = resolve_program("./does-not-exist-abc").unwrap_err();
        assert!(err.to_string().contains("command not found"));
    }

    #[test]
    fn resolve_program_bare_name_not_on_path() {
        let err =
            resolve_program("definitely-not-a-real-binary-xyzzy-987").unwrap_err();
        assert!(err.to_string().contains("command not found"));
    }

    #[test]
    fn resolve_program_bare_name_found_on_path() {
        // /bin/sh exists on every supported platform (macOS + Linux).
        let resolved = resolve_program("sh").expect("sh should be on PATH");
        assert!(resolved.is_absolute(), "got: {resolved:?}");
        assert_eq!(resolved.file_name().and_then(|s| s.to_str()), Some("sh"));
    }
}

/// Start a daemon without attaching (useful for scripts/tests)
fn cmd_start(name: Option<String>, command: Vec<String>) -> Result<()> {
    if command.is_empty() {
        anyhow::bail!("no command specified");
    }

    resolve_program(&command[0])?;

    let session_name = name.unwrap_or_else(|| {
        session::generate_unique_name().unwrap_or_else(|_| session::generate_name())
    });

    if session::load_session(&session_name)?.is_some() {
        anyhow::bail!("session '{}' already exists", session_name);
    }

    daemon::start_daemon(session_name.clone(), command)?;

    println!("{}", session_name);

    Ok(())
}
