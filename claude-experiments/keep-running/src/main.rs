mod cli;
mod client;
mod daemon;
mod error;
mod protocol;
mod session;
mod terminal;

use anyhow::{Context, Result};
use cli::Commands;

fn main() -> Result<()> {
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
    // Try SHELL environment variable first
    if let Ok(shell) = std::env::var("SHELL") {
        return shell;
    }

    // Fallback to /bin/sh
    "/bin/sh".to_string()
}

fn cmd_shell(name: Option<String>) -> Result<()> {
    let shell = detect_shell();
    let command = vec![shell];
    cmd_run(name, command)
}

fn cmd_run(name: Option<String>, command: Vec<String>) -> Result<()> {
    if command.is_empty() {
        anyhow::bail!("No command specified");
    }

    let session_name = name.unwrap_or_else(|| {
        session::generate_unique_name().unwrap_or_else(|_| session::generate_name())
    });

    // Check if session already exists
    if session::load_session(&session_name)?.is_some() {
        anyhow::bail!("Session '{}' already exists", session_name);
    }

    println!("Starting session: {}", session_name);
    println!("Detach: Ctrl+a d | Kill: Ctrl+a k");
    println!();

    client::run_and_attach(&session_name, &command)?;

    Ok(())
}

fn cmd_attach(session_query: &str) -> Result<()> {
    let session = session::find_session(session_query)?
        .with_context(|| format!("Session '{}' not found", session_query))?;

    println!("Attaching to: {}", session.name);
    println!("Detach: Ctrl+a d | Kill: Ctrl+a k");
    println!();

    client::attach(&session)?;

    Ok(())
}

fn cmd_list() -> Result<()> {
    let sessions = session::list_sessions()?;

    if sessions.is_empty() {
        println!("No running sessions");
        return Ok(());
    }

    println!("{:<20} {:<10} {}", "NAME", "PID", "COMMAND");
    println!("{}", "-".repeat(60));

    for session in sessions {
        let cmd = session.command.join(" ");
        let cmd_display = if cmd.len() > 30 {
            format!("{}...", &cmd[..27])
        } else {
            cmd
        };
        println!("{:<20} {:<10} {}", session.name, session.pid, cmd_display);
    }

    Ok(())
}

fn cmd_kill(session_query: &str) -> Result<()> {
    let session = session::find_session(session_query)?
        .with_context(|| format!("Session '{}' not found", session_query))?;

    // Send SIGTERM to the daemon
    unsafe {
        libc::kill(session.pid as i32, libc::SIGTERM);
    }

    // Clean up session files
    session::remove_session(&session.name)?;

    println!("Killed session: {}", session.name);

    Ok(())
}

/// Start a daemon without attaching (useful for scripts/tests)
fn cmd_start(name: Option<String>, command: Vec<String>) -> Result<()> {
    if command.is_empty() {
        anyhow::bail!("No command specified");
    }

    let session_name = name.unwrap_or_else(|| {
        session::generate_unique_name().unwrap_or_else(|_| session::generate_name())
    });

    // Check if session already exists
    if session::load_session(&session_name)?.is_some() {
        anyhow::bail!("Session '{}' already exists", session_name);
    }

    // Start the daemon
    daemon::start_daemon(session_name.clone(), command)?;

    println!("{}", session_name);

    Ok(())
}
