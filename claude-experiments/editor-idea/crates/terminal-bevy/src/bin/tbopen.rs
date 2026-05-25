//! `tbopen` — like `subl` but for the running `terminal-bevy` app.
//!
//! Sends a single JSON line over the app's Unix socket asking it to
//! open a file in an editor pane. The app must already be running
//! (we don't auto-launch it).
//!
//! Usage:
//!     tbopen <file> [--project <name>]
//!
//! Default project is whichever project is currently active in the
//! running app. `--project NAME` does a case-insensitive match against
//! the project list; if no project matches the request is silently
//! dropped (the app logs to stderr).
//!
//! The wire format is duplicated here on purpose: the parent
//! `terminal_bevy` lib depends on libghostty-vt (a dylib), and a bin
//! in the same package links against the lib by default. Keeping this
//! bin lib-free means no @rpath dance to ship `tbopen` alongside the
//! main app.

use std::io::Write;
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use serde::Serialize;

#[derive(Serialize)]
#[serde(tag = "action", rename_all = "snake_case")]
enum IpcRequest {
    OpenFile {
        path: PathBuf,
        #[serde(skip_serializing_if = "Option::is_none")]
        project: Option<String>,
    },
}

fn socket_path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    Some(Path::new(&home).join(".terminal-bevy").join("socket"))
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

    let abs = match args.path.canonicalize() {
        Ok(p) => p,
        Err(e) => {
            eprintln!("tbopen: {}: {}", args.path.display(), e);
            return ExitCode::from(1);
        }
    };

    let Some(sock) = socket_path() else {
        eprintln!("tbopen: $HOME not set; can't locate socket");
        return ExitCode::from(1);
    };

    let mut stream = match UnixStream::connect(&sock) {
        Ok(s) => s,
        Err(e) => {
            eprintln!(
                "tbopen: connect {}: {} (is the terminal-bevy app running?)",
                sock.display(),
                e
            );
            return ExitCode::from(1);
        }
    };

    let req = IpcRequest::OpenFile {
        path: abs,
        project: args.project,
    };
    let body = match serde_json::to_vec(&req) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("tbopen: serialize: {}", e);
            return ExitCode::from(1);
        }
    };
    if let Err(e) = stream.write_all(&body) {
        eprintln!("tbopen: write: {}", e);
        return ExitCode::from(1);
    }
    // Half-close so the app sees EOF and parses our message.
    let _ = stream.shutdown(std::net::Shutdown::Write);
    ExitCode::SUCCESS
}

struct Args {
    path: PathBuf,
    project: Option<String>,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut path: Option<PathBuf> = None;
        let mut project: Option<String> = None;
        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "-h" | "--help" => {
                    print_usage();
                    std::process::exit(0);
                }
                "--project" | "-p" => {
                    project = Some(
                        it.next()
                            .ok_or_else(|| format!("{} requires a value", arg))?,
                    );
                }
                other if other.starts_with("--") => {
                    return Err(format!("unknown flag: {}", other));
                }
                other => {
                    if path.is_some() {
                        return Err(format!("unexpected extra arg: {}", other));
                    }
                    path = Some(PathBuf::from(other));
                }
            }
        }
        Ok(Self {
            path: path.ok_or("missing <file> argument")?,
            project,
        })
    }
}

fn print_usage() {
    eprintln!(
        "tbopen <file> [--project NAME]\n\
         \n\
         Send <file> to the running terminal-bevy app, which opens it in a\n\
         new editor pane. Default project is whichever is currently active."
    );
}
