//! `tbmsg` — talk to the widget↔widget message bus from the shell.
//!
//! The bus lets widget panes in the same editor project coordinate (an
//! editor pane tells a results pane "run this query", the results pane
//! tells everyone "query finished", etc.). This CLI is the shell-side
//! door into it — handy for driving a widget from a `proc_spawn`ed child
//! or verifying message flow without the GUI. It mirrors `claude-bus-tail`.
//!
//! Usage:
//!   tbmsg emit --project P --topic T [--json '{...}'] [--retain]
//!   tbmsg tail [--project P]
//!
//!   emit   Publish one message. Delivered to every widget in project P
//!          as `on_message(topic, payload, "tbmsg")`. `--retain` keeps it
//!          as the topic's last value for widgets that spawn later.
//!   tail   Follow the bus live, printing each delivered message as a
//!          JSON line. `--project P` filters to that project.
//!
//! `--project` accepts a project name (`datalog-db`) or `active`.

use std::io::{BufRead, BufReader, Read, Seek, SeekFrom, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

fn socket_path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    Some(Path::new(&home).join(".terminal-bevy").join("socket"))
}

fn bus_log_path() -> Option<PathBuf> {
    let home = std::env::var_os("HOME")?;
    Some(Path::new(&home).join(".terminal-bevy").join("widget-bus.log"))
}

fn print_usage() {
    eprintln!(
        "usage:\n  \
         tbmsg emit --project P --topic T [--json '{{...}}'] [--retain]\n  \
         tbmsg tail [--project P]"
    );
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some(sub) = args.first() else {
        print_usage();
        return ExitCode::from(2);
    };
    match sub.as_str() {
        "emit" => cmd_emit(&args[1..]),
        "tail" => cmd_tail(&args[1..]),
        "-h" | "--help" | "help" => {
            print_usage();
            ExitCode::SUCCESS
        }
        other => {
            eprintln!("tbmsg: unknown subcommand `{}`", other);
            print_usage();
            ExitCode::from(2)
        }
    }
}

/// Pull `--flag value` / `--flag=value` pairs and bare `--flag` switches
/// out of an argv slice. Returns (named, switches).
fn parse_flags(args: &[String]) -> (Vec<(String, String)>, Vec<String>) {
    let mut named = Vec::new();
    let mut switches = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        if let Some(rest) = a.strip_prefix("--") {
            if let Some((k, v)) = rest.split_once('=') {
                named.push((k.to_string(), v.to_string()));
            } else if i + 1 < args.len() && !args[i + 1].starts_with("--") {
                named.push((rest.to_string(), args[i + 1].clone()));
                i += 1;
            } else {
                switches.push(rest.to_string());
            }
        }
        i += 1;
    }
    (named, switches)
}

fn get<'a>(named: &'a [(String, String)], key: &str) -> Option<&'a str> {
    named.iter().find(|(k, _)| k == key).map(|(_, v)| v.as_str())
}

fn cmd_emit(args: &[String]) -> ExitCode {
    let (named, switches) = parse_flags(args);
    let Some(topic) = get(&named, "topic") else {
        eprintln!("tbmsg emit: --topic is required");
        print_usage();
        return ExitCode::from(2);
    };
    let payload: serde_json::Value = match get(&named, "json") {
        Some(raw) => match serde_json::from_str(raw) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("tbmsg emit: --json is not valid JSON: {}", e);
                return ExitCode::from(2);
            }
        },
        None => serde_json::Value::Null,
    };
    let retain = switches.iter().any(|s| s == "retain");

    let req = serde_json::json!({
        "action": "widget_message",
        "project": get(&named, "project"),
        "topic": topic,
        "payload": payload,
        "retain": retain,
    });

    let Some(sock) = socket_path() else {
        eprintln!("tbmsg: $HOME not set; can't locate socket");
        return ExitCode::from(1);
    };
    let mut stream = match UnixStream::connect(&sock) {
        Ok(s) => s,
        Err(e) => {
            eprintln!(
                "tbmsg: connect {}: {} (is the terminal-bevy app running?)",
                sock.display(),
                e
            );
            return ExitCode::from(1);
        }
    };
    let body = match serde_json::to_vec(&req) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("tbmsg: serialize: {}", e);
            return ExitCode::from(1);
        }
    };
    if let Err(e) = stream.write_all(&body) {
        eprintln!("tbmsg: write: {}", e);
        return ExitCode::from(1);
    }
    let _ = stream.shutdown(std::net::Shutdown::Write);
    ExitCode::SUCCESS
}

fn cmd_tail(args: &[String]) -> ExitCode {
    let (named, _switches) = parse_flags(args);
    // Resolve `--project NAME` to its numeric id (the log stores ids) by
    // asking the running app. Absent → show every project.
    let filter_id: Option<u64> = match get(&named, "project") {
        Some(name) => match resolve_project_id(name) {
            Ok(id) => Some(id),
            Err(e) => {
                eprintln!("tbmsg tail: {}", e);
                return ExitCode::from(1);
            }
        },
        None => None,
    };

    let Some(log) = bus_log_path() else {
        eprintln!("tbmsg: $HOME not set; can't locate bus log");
        return ExitCode::from(1);
    };

    // Follow the log like `tail -f`: seek to the end, then poll for newly
    // appended lines. If the file is truncated (app restart), reset.
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    let mut pos: u64 = std::fs::metadata(&log).map(|m| m.len()).unwrap_or(0);
    loop {
        let mut f = match std::fs::File::open(&log) {
            Ok(f) => f,
            Err(_) => {
                std::thread::sleep(std::time::Duration::from_millis(300));
                continue;
            }
        };
        let len = f.metadata().map(|m| m.len()).unwrap_or(0);
        if len < pos {
            // Truncated (restart) — start over from the top.
            pos = 0;
        }
        if len > pos {
            if f.seek(SeekFrom::Start(pos)).is_err() {
                pos = len;
                continue;
            }
            let mut reader = BufReader::new(&mut f);
            let mut line = String::new();
            loop {
                line.clear();
                match reader.read_line(&mut line) {
                    Ok(0) => break,
                    Ok(_) => {
                        if line.ends_with('\n') {
                            print_tail_line(&mut out, line.trim_end(), filter_id);
                        } else {
                            // Partial line: rewind so we re-read it whole.
                            break;
                        }
                    }
                    Err(_) => break,
                }
            }
            // Where the next read should resume from.
            pos = f.stream_position().unwrap_or(len);
            let _ = out.flush();
        }
        std::thread::sleep(std::time::Duration::from_millis(200));
    }
}

fn print_tail_line(out: &mut impl Write, line: &str, filter_id: Option<u64>) {
    if line.is_empty() {
        return;
    }
    if let Some(want) = filter_id {
        // Each line is `{"project":<id|null>,...}`; filter by project id.
        let parsed: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => return,
        };
        let pid = parsed.get("project").and_then(|v| v.as_u64());
        if pid != Some(want) {
            return;
        }
    }
    let _ = writeln!(out, "{}", line);
}

/// Ask the running app for its project list and resolve `name` to an id.
fn resolve_project_id(name: &str) -> Result<u64, String> {
    let sock = socket_path().ok_or_else(|| "$HOME not set".to_string())?;
    let mut stream = UnixStream::connect(&sock)
        .map_err(|e| format!("connect {}: {} (is the app running?)", sock.display(), e))?;
    stream
        .write_all(br#"{"action":"list_projects"}"#)
        .map_err(|e| format!("write: {}", e))?;
    let _ = stream.shutdown(std::net::Shutdown::Write);
    let mut body = String::new();
    stream
        .read_to_string(&mut body)
        .map_err(|e| format!("read: {}", e))?;
    let parsed: serde_json::Value =
        serde_json::from_str(&body).map_err(|e| format!("bad response: {}", e))?;
    let projects = parsed
        .get("projects")
        .and_then(|v| v.as_array())
        .ok_or_else(|| "no projects in response".to_string())?;
    for p in projects {
        let pname = p.get("name").and_then(|v| v.as_str()).unwrap_or("");
        if pname.eq_ignore_ascii_case(name) {
            return p
                .get("id")
                .and_then(|v| v.as_u64())
                .ok_or_else(|| "project has no id".to_string());
        }
    }
    Err(format!("no project named `{}`", name))
}
