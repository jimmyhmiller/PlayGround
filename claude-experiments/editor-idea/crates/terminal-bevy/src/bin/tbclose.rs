//! `tbclose` — close (despawn) panes in a project from the shell.
//!
//! Routes through the normal pane-close path (the kind's `on_close` +
//! despawn), i.e. the scriptable equivalent of clicking each pane's close
//! button. Handy for cleaning up panes spawned via `tbwidget`/`tbopen`.
//!
//! Usage:
//!   tbclose --project P [--kind K]
//!
//!   --project P   project name (or `active`). Required.
//!   --kind K      only close panes of this kind (e.g. `rhai_widget`,
//!                 `widget`, `editor`). Omit to close EVERY pane in P.

use std::io::Write;
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

fn socket_path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    Some(Path::new(&home).join(".jim").join("socket"))
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut project: Option<String> = None;
    let mut kind: Option<String> = None;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--project" | "-p" => {
                project = args.get(i + 1).cloned();
                i += 1;
            }
            "--kind" | "-k" => {
                kind = args.get(i + 1).cloned();
                i += 1;
            }
            "-h" | "--help" => {
                eprintln!("usage: tbclose --project P [--kind K]");
                return ExitCode::SUCCESS;
            }
            other => {
                eprintln!("tbclose: unexpected arg `{}`", other);
                eprintln!("usage: tbclose --project P [--kind K]");
                return ExitCode::from(2);
            }
        }
        i += 1;
    }
    if project.is_none() {
        eprintln!("tbclose: --project is required");
        return ExitCode::from(2);
    }

    let req = serde_json::json!({
        "action": "close_project_panes",
        "project": project,
        "kind": kind,
    });

    let Some(sock) = socket_path() else {
        eprintln!("tbclose: $HOME not set; can't locate socket");
        return ExitCode::from(1);
    };
    let mut stream = match UnixStream::connect(&sock) {
        Ok(s) => s,
        Err(e) => {
            eprintln!(
                "tbclose: connect {}: {} (is the terminal-bevy app running?)",
                sock.display(),
                e
            );
            return ExitCode::from(1);
        }
    };
    let body = match serde_json::to_vec(&req) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("tbclose: serialize: {}", e);
            return ExitCode::from(1);
        }
    };
    if let Err(e) = stream.write_all(&body) {
        eprintln!("tbclose: write: {}", e);
        return ExitCode::from(1);
    }
    let _ = stream.shutdown(std::net::Shutdown::Write);
    ExitCode::SUCCESS
}
