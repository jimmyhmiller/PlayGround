//! `tbwidget` — spawn a new widget pane in the running `terminal-bevy`
//! app. Mirrors `tbopen`'s socket dance.
//!
//! Usage:
//!     tbwidget [--title T] [--cwd D] [--project P] -- <cmd> [args...]
//!     tbwidget [--title T] [--cwd D] [--project P] <cmd-with-spaces>
//!
//! Two argv shapes are accepted:
//!   - Everything after `--` is taken as `argv` for the child (no shell).
//!     Example: `tbwidget --title issues -- gh-issues.sh`.
//!   - No `--` → the remaining single positional arg is the command line
//!     and is run through `sh -c`. Example:
//!     `tbwidget --title issues "gh issue list | jq -c '...'"`.
//!
//! `--cwd` defaults to the caller's current directory so relative paths
//! and scripts that read `$PWD` keep working.
//!
//! Wire format is duplicated here on purpose (same rationale as
//! `tbopen`): a same-package bin links the lib's dylib transitively
//! and we don't want to ship the @rpath dance with this CLI.

use std::io::Write;
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

use serde::Serialize;

#[derive(Serialize)]
#[serde(tag = "action", rename_all = "snake_case")]
enum IpcRequest {
    SpawnWidget {
        command: String,
        #[serde(skip_serializing_if = "Vec::is_empty")]
        args: Vec<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        title: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        cwd: Option<PathBuf>,
        #[serde(skip_serializing_if = "Option::is_none")]
        project: Option<String>,
        /// Optional widget kind override. Default is the subprocess
        /// widget kind. Pass `"rhai_widget"` to spawn an in-process
        /// Rhai-scripted widget; `command` is then the script filename
        /// under `~/.terminal-bevy/widgets/`.
        #[serde(skip_serializing_if = "Option::is_none")]
        kind: Option<String>,
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

    let Some(sock) = socket_path() else {
        eprintln!("tbwidget: $HOME not set; can't locate socket");
        return ExitCode::from(1);
    };

    let mut stream = match UnixStream::connect(&sock) {
        Ok(s) => s,
        Err(e) => {
            eprintln!(
                "tbwidget: connect {}: {} (is the terminal-bevy app running?)",
                sock.display(),
                e
            );
            return ExitCode::from(1);
        }
    };

    let req = IpcRequest::SpawnWidget {
        command: args.command,
        args: args.args,
        title: args.title,
        cwd: args.cwd,
        project: args.project,
        kind: args.kind,
    };
    let body = match serde_json::to_vec(&req) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("tbwidget: serialize: {}", e);
            return ExitCode::from(1);
        }
    };
    if let Err(e) = stream.write_all(&body) {
        eprintln!("tbwidget: write: {}", e);
        return ExitCode::from(1);
    }
    let _ = stream.shutdown(std::net::Shutdown::Write);
    ExitCode::SUCCESS
}

struct Args {
    command: String,
    args: Vec<String>,
    title: Option<String>,
    cwd: Option<PathBuf>,
    project: Option<String>,
    kind: Option<String>,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut title: Option<String> = None;
        let mut cwd: Option<PathBuf> = None;
        let mut project: Option<String> = None;
        let mut kind: Option<String> = None;
        let mut positional: Vec<String> = Vec::new();
        let mut argv_mode = false;
        let mut argv_after_dash: Vec<String> = Vec::new();

        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            if argv_mode {
                argv_after_dash.push(arg);
                continue;
            }
            match arg.as_str() {
                "-h" | "--help" => {
                    print_usage();
                    std::process::exit(0);
                }
                "--title" | "-t" => {
                    title = Some(
                        it.next()
                            .ok_or_else(|| format!("{} requires a value", arg))?,
                    );
                }
                "--cwd" => {
                    cwd = Some(PathBuf::from(
                        it.next()
                            .ok_or_else(|| format!("{} requires a value", arg))?,
                    ));
                }
                "--project" | "-p" => {
                    project = Some(
                        it.next()
                            .ok_or_else(|| format!("{} requires a value", arg))?,
                    );
                }
                "--kind" | "-k" => {
                    kind = Some(
                        it.next()
                            .ok_or_else(|| format!("{} requires a value", arg))?,
                    );
                }
                "--" => {
                    argv_mode = true;
                }
                other if other.starts_with("--") => {
                    return Err(format!("unknown flag: {}", other));
                }
                other => {
                    positional.push(other.into());
                }
            }
        }

        // Default cwd to caller's PWD so scripts behave as if launched
        // from the shell that ran us.
        if cwd.is_none() {
            cwd = std::env::current_dir().ok();
        }

        let (command, child_args) = if !argv_after_dash.is_empty() {
            let mut it = argv_after_dash.into_iter();
            let head = it.next().expect("non-empty after `--`");
            let rest: Vec<String> = it.collect();
            (head, rest)
        } else if positional.len() == 1 {
            // Single positional → shell command line.
            (positional.into_iter().next().unwrap(), Vec::new())
        } else if positional.is_empty() {
            return Err("missing command — pass `-- <cmd>` or a single quoted shell line".into());
        } else {
            return Err(format!(
                "got {} positional args without `--`; use `-- {} ...` to pass them as argv",
                positional.len(),
                positional[0],
            ));
        };

        Ok(Self {
            command,
            args: child_args,
            title,
            cwd,
            project,
            kind,
        })
    }
}

fn print_usage() {
    eprintln!(
        "tbwidget [--title T] [--cwd D] [--project P] [--kind K] -- <cmd> [args...]\n\
         tbwidget [--title T] [--cwd D] [--project P] [--kind K] <shell-line>\n\
         \n\
         Spawn a new widget pane in the running terminal-bevy app. The\n\
         child speaks the widget NDJSON protocol over stdout/stdin.\n\
         \n\
         With `--`, the remaining args become argv and the child runs\n\
         directly. Without `--`, a single quoted positional is passed to\n\
         `sh -c`.\n\
         \n\
         `--kind rhai_widget` swaps in the in-process Rhai-scripted\n\
         widget runtime. `<cmd>` is then interpreted as a script filename\n\
         under `~/.terminal-bevy/widgets/` and no subprocess is spawned.\n\
         \n\
         For the authoring guide, run `widget agent`."
    );
}
