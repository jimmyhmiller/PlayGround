//! `tbinbox` — push a message into the running terminal-bevy app's
//! per-project inbox over its Unix socket.
//!
//! Usage:
//!     tbinbox --body "hello" [--project NAME] [--sender X] [--subject Y]
//!     echo "stdin body" | tbinbox --project alpha
//!
//! `--project` defaults to whichever project is currently active. The
//! body can be passed via `--body` OR piped on stdin (stdin wins when
//! both are present).
//!
//! Like `tbopen`, this binary deliberately stays lib-free: the parent
//! `terminal_bevy` crate links a dylib (libghostty-vt) we don't want
//! to pull into a tiny CLI.

use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use serde::Serialize;

#[derive(Serialize)]
#[serde(tag = "action", rename_all = "snake_case")]
enum IpcRequest {
    SendInbox {
        #[serde(skip_serializing_if = "Option::is_none")]
        project: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        sender: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        subject: Option<String>,
        body: String,
    },
}

fn socket_path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    Some(Path::new(&home).join(".jim").join("socket"))
}

fn print_usage() {
    eprintln!(
        "usage: tbinbox [--project NAME] [--sender X] [--subject Y] (--body TEXT | < stdin)"
    );
}

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1).collect::<Vec<_>>().into_iter();
    let mut project: Option<String> = None;
    let mut sender: Option<String> = None;
    let mut subject: Option<String> = None;
    let mut body_arg: Option<String> = None;

    while let Some(a) = args.next() {
        match a.as_str() {
            "--project" => project = args.next(),
            "--sender" => sender = args.next(),
            "--subject" => subject = args.next(),
            "--body" => body_arg = args.next(),
            "-h" | "--help" => {
                print_usage();
                return ExitCode::SUCCESS;
            }
            other => {
                eprintln!("tbinbox: unknown arg {:?}", other);
                print_usage();
                return ExitCode::from(2);
            }
        }
    }

    // Stdin wins over --body when stdin is piped (not a tty).
    let body = match read_stdin_if_piped() {
        Some(s) if !s.is_empty() => s,
        _ => match body_arg {
            Some(s) => s,
            None => {
                eprintln!("tbinbox: need --body TEXT or piped stdin");
                print_usage();
                return ExitCode::from(2);
            }
        },
    };

    let Some(path) = socket_path() else {
        eprintln!("tbinbox: $HOME not set");
        return ExitCode::from(1);
    };

    let mut sock = match UnixStream::connect(&path) {
        Ok(s) => s,
        Err(e) => {
            eprintln!(
                "tbinbox: connect {}: {}\n  (is terminal-bevy running?)",
                path.display(),
                e
            );
            return ExitCode::from(1);
        }
    };

    let req = IpcRequest::SendInbox {
        project,
        sender,
        subject,
        body,
    };
    let bytes = match serde_json::to_vec(&req) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("tbinbox: serialize: {}", e);
            return ExitCode::from(1);
        }
    };
    if let Err(e) = sock.write_all(&bytes) {
        eprintln!("tbinbox: write: {}", e);
        return ExitCode::from(1);
    }
    // EOF tells the app side we're done — it expects single-shot
    // JSON-to-EOF on each connection.
    let _ = sock.shutdown(std::net::Shutdown::Write);
    ExitCode::SUCCESS
}

fn read_stdin_if_piped() -> Option<String> {
    use std::io::IsTerminal;
    let stdin = std::io::stdin();
    if stdin.is_terminal() {
        return None;
    }
    let mut buf = String::new();
    let mut handle = stdin.lock();
    handle.read_to_string(&mut buf).ok()?;
    Some(buf.trim_end_matches('\n').to_string())
}
