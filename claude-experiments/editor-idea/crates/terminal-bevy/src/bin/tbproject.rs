//! `tbproject` — manage projects in the running terminal-bevy app
//! over its Unix socket.
//!
//! Today there's one subcommand: `set-cwd`, which writes a project's
//! `default_cwd` (the directory new terminals are spawned in for that
//! project, also the cwd a Claude widget falls back to). Without an
//! explicit path, the current shell cwd is used.
//!
//! Usage:
//!     tbproject set-cwd                     # active project ← $PWD
//!     tbproject set-cwd .                   # same, explicit
//!     tbproject set-cwd /some/path
//!     tbproject set-cwd --project Recursion /some/path
//!     tbproject set-cwd --project Recursion --clear
//!
//! Stays lib-free for the same reason as `tbopen` / `tbinbox` — we don't
//! want to drag libghostty-vt into a tiny CLI.

use std::io::Write;
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use serde::Serialize;

#[derive(Serialize)]
#[serde(tag = "action", rename_all = "snake_case")]
enum IpcRequest {
    SetProjectDefaultCwd {
        #[serde(skip_serializing_if = "Option::is_none")]
        project: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        cwd: Option<PathBuf>,
    },
}

fn socket_path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    Some(Path::new(&home).join(".terminal-bevy").join("socket"))
}

fn print_usage() {
    eprintln!(
        "usage: tbproject set-cwd [--project NAME] [PATH | --clear]\n\
         \n\
         Sets a project's default_cwd. PATH defaults to the current\n\
         shell directory. --project defaults to the active project.\n\
         --clear removes the override so new terminals fall back to $HOME."
    );
}

fn main() -> ExitCode {
    let mut args = std::env::args().skip(1);
    let Some(sub) = args.next() else {
        print_usage();
        return ExitCode::from(2);
    };
    match sub.as_str() {
        "set-cwd" => set_cwd(args),
        "-h" | "--help" => {
            print_usage();
            ExitCode::SUCCESS
        }
        other => {
            eprintln!("tbproject: unknown subcommand {:?}", other);
            print_usage();
            ExitCode::from(2)
        }
    }
}

fn set_cwd(args: impl Iterator<Item = String>) -> ExitCode {
    let mut project: Option<String> = None;
    let mut path_arg: Option<String> = None;
    let mut clear = false;
    let mut it = args.peekable();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--project" => match it.next() {
                Some(v) => project = Some(v),
                None => {
                    eprintln!("tbproject: --project needs a value");
                    return ExitCode::from(2);
                }
            },
            "--clear" => clear = true,
            "-h" | "--help" => {
                print_usage();
                return ExitCode::SUCCESS;
            }
            other if other.starts_with("--") => {
                eprintln!("tbproject set-cwd: unknown flag {:?}", other);
                return ExitCode::from(2);
            }
            _ => {
                if path_arg.is_some() {
                    eprintln!("tbproject set-cwd: extra positional arg {:?}", a);
                    return ExitCode::from(2);
                }
                path_arg = Some(a);
            }
        }
    }

    if clear && path_arg.is_some() {
        eprintln!("tbproject set-cwd: --clear can't be combined with a PATH");
        return ExitCode::from(2);
    }

    let cwd: Option<PathBuf> = if clear {
        None
    } else {
        let raw = path_arg.unwrap_or_else(|| ".".to_string());
        let p = PathBuf::from(&raw);
        let abs = if p.is_absolute() {
            p
        } else {
            match std::env::current_dir() {
                Ok(cwd) => cwd.join(&p),
                Err(e) => {
                    eprintln!("tbproject set-cwd: getcwd: {}", e);
                    return ExitCode::from(1);
                }
            }
        };
        // Canonicalize when possible so the app stores a stable path,
        // but fall back to the joined form if the directory doesn't
        // exist yet (matches macOS Terminal's "open in" behavior).
        let canon = std::fs::canonicalize(&abs).unwrap_or(abs);
        Some(canon)
    };

    let Some(socket) = socket_path() else {
        eprintln!("tbproject: $HOME not set");
        return ExitCode::from(1);
    };

    let mut sock = match UnixStream::connect(&socket) {
        Ok(s) => s,
        Err(e) => {
            eprintln!(
                "tbproject: connect {}: {}\n  (is terminal-bevy running?)",
                socket.display(),
                e
            );
            return ExitCode::from(1);
        }
    };

    let req = IpcRequest::SetProjectDefaultCwd {
        project: project.clone(),
        cwd: cwd.clone(),
    };
    let bytes = match serde_json::to_vec(&req) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("tbproject: serialize: {}", e);
            return ExitCode::from(1);
        }
    };
    if let Err(e) = sock.write_all(&bytes) {
        eprintln!("tbproject: write: {}", e);
        return ExitCode::from(1);
    }
    let _ = sock.shutdown(std::net::Shutdown::Write);

    match (&project, &cwd) {
        (Some(p), Some(c)) => println!("tbproject: project {:?} default_cwd ← {}", p, c.display()),
        (None, Some(c)) => println!("tbproject: active project default_cwd ← {}", c.display()),
        (Some(p), None) => println!("tbproject: project {:?} default_cwd cleared", p),
        (None, None) => println!("tbproject: active project default_cwd cleared"),
    }
    ExitCode::SUCCESS
}
