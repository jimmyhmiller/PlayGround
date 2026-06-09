//! `tbsuggest` — park a *suggested* pane in the running app's drawer
//! (the Quake-style dropdown), instead of spawning it on the canvas.
//!
//! The AI calls this when it infers a pane might be useful but doesn't
//! want to clutter the surface — e.g. it just ran a command in a side
//! terminal and thinks you might want a run-button for it. The user
//! pulls the drawer down (Cmd+J) later and picks it.
//!
//! Usage:
//!     tbsuggest --command "cargo test" [--title "Run tests"] [--cwd DIR]
//!               [--reason "you just ran this in a side terminal"]
//!               [--project NAME]
//!     tbsuggest --kind editor --config '{"path":"/abs/file.rs"}' --title "open foo"
//!
//! With a bare `--command` and no `--kind`, the suggestion becomes a
//! `run-button` pre-filled with the command (and `--cwd`/`--title`).
//! For any other pane kind, pass `--kind` plus an explicit `--config`
//! JSON blob (the same shape that kind persists in `projects.json`).
//!
//! The app must already be running. The wire format is duplicated here
//! on purpose so this bin stays free of the libghostty-vt dylib (see
//! `tbopen.rs` for the rationale).

use std::io::Write;
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use serde::Serialize;

#[derive(Serialize)]
#[serde(tag = "action", rename_all = "snake_case")]
enum IpcRequest {
    SuggestPane {
        #[serde(skip_serializing_if = "Option::is_none")]
        kind: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        title: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        command: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        cwd: Option<PathBuf>,
        #[serde(skip_serializing_if = "Option::is_none")]
        reason: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        config: Option<serde_json::Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        project: Option<String>,
        /// The dir tbsuggest was invoked in. The app maps it to the
        /// owning project (by `default_cwd`) when `project` isn't given,
        /// so a suggestion fired from a side terminal lands in that
        /// terminal's project, not whatever's active in the GUI.
        #[serde(skip_serializing_if = "Option::is_none")]
        from_cwd: Option<PathBuf>,
    },
}

fn socket_path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    Some(Path::new(&home).join(".jim").join("socket"))
}

fn main() -> ExitCode {
    let args = match Args::parse() {
        Ok(a) => a,
        Err(msg) => {
            eprintln!("{}", msg);
            print_usage();
            return ExitCode::from(2);
        }
    };

    if args.kind.is_none() && args.command.is_none() {
        eprintln!("tbsuggest: need --command or --kind");
        print_usage();
        return ExitCode::from(2);
    }

    let config = match &args.config {
        Some(raw) => match serde_json::from_str::<serde_json::Value>(raw) {
            Ok(v) => Some(v),
            Err(e) => {
                eprintln!("tbsuggest: --config is not valid JSON: {}", e);
                return ExitCode::from(2);
            }
        },
        None => None,
    };

    // Resolve --cwd to an absolute path so the materialized pane runs in
    // the dir the caller meant, not the app's cwd.
    let cwd = match args.cwd {
        Some(p) => match p.canonicalize() {
            Ok(abs) => Some(abs),
            Err(_) => Some(p), // pass through; the app reports if it's bad
        },
        None => None,
    };

    let Some(sock) = socket_path() else {
        eprintln!("tbsuggest: $HOME not set; can't locate socket");
        return ExitCode::from(1);
    };

    let mut stream = match UnixStream::connect(&sock) {
        Ok(s) => s,
        Err(e) => {
            eprintln!(
                "tbsuggest: connect {}: {} (is the terminal-bevy app running?)",
                sock.display(),
                e
            );
            return ExitCode::from(1);
        }
    };

    let from_cwd = std::env::current_dir().ok();

    let req = IpcRequest::SuggestPane {
        kind: args.kind,
        title: args.title,
        command: args.command,
        cwd,
        reason: args.reason,
        config,
        project: args.project,
        from_cwd,
    };
    let body = match serde_json::to_vec(&req) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("tbsuggest: serialize: {}", e);
            return ExitCode::from(1);
        }
    };
    if let Err(e) = stream.write_all(&body) {
        eprintln!("tbsuggest: write: {}", e);
        return ExitCode::from(1);
    }
    let _ = stream.shutdown(std::net::Shutdown::Write);
    ExitCode::SUCCESS
}

#[derive(Default)]
struct Args {
    kind: Option<String>,
    title: Option<String>,
    command: Option<String>,
    cwd: Option<PathBuf>,
    reason: Option<String>,
    config: Option<String>,
    project: Option<String>,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut a = Args::default();
        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            let mut take = |name: &str| -> Result<String, String> {
                it.next().ok_or_else(|| format!("{} requires a value", name))
            };
            match arg.as_str() {
                "-h" | "--help" => {
                    print_usage();
                    std::process::exit(0);
                }
                "--kind" => a.kind = Some(take("--kind")?),
                "--title" | "-t" => a.title = Some(take("--title")?),
                "--command" | "-c" => a.command = Some(take("--command")?),
                "--cwd" => a.cwd = Some(PathBuf::from(take("--cwd")?)),
                "--reason" | "-r" => a.reason = Some(take("--reason")?),
                "--config" => a.config = Some(take("--config")?),
                "--project" | "-p" => a.project = Some(take("--project")?),
                other => return Err(format!("unknown argument: {}", other)),
            }
        }
        Ok(a)
    }
}

fn print_usage() {
    eprintln!(
        "tbsuggest --command CMD [--title T] [--cwd DIR] [--reason R] [--project NAME]\n\
         tbsuggest --kind KIND --config JSON [--title T] [--reason R] [--project NAME]\n\
         \n\
         Park a suggested pane in the running app's drawer (Cmd+J to open).\n\
         A bare --command becomes a run-button; other kinds need --config.\n\
         The user pulls the drawer down and picks it to spawn on the canvas."
    );
}
