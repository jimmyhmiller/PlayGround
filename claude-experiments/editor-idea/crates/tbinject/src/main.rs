//! `tbinject` — send keystrokes into a running terminal-bevy session.
//!
//! Connects to the daemon's inject side-channel (`<id>.inject`),
//! writes the requested bytes, closes. The PTY treats the bytes as
//! though the user typed them — same path as a Claude Code prompt
//! submission, an `Esc` to cancel generation, etc.
//!
//! Usage:
//!     tbinject --session 27 --text "hello\n"
//!     tbinject --project beagle --text "/compact\n"
//!     tbinject --session 27 --bytes 1b           # raw byte 0x1b (Esc)
//!     tbinject --session 27 --text "..." --text "..."  # multiple frames
//!
//! Session/project resolution mirrors `tbwidget`: `--session <id>` is
//! the literal numeric `TerminalSession`; `--project <name>` picks
//! the most-recently-created terminal in that project from
//! `~/.jim/terminals.json`.
//!
//! Bytes flag: hex pairs, no separators (`1b`, `0d0a`, `030d`).
//! Useful for sending Esc, Ctrl-C (`03`), Enter (`0d`), etc.

use std::io::{Read, Write};
use std::os::unix::net::UnixStream;
use std::process::ExitCode;

use serde::Deserialize;

#[derive(Deserialize)]
struct LiveTerminals {
    #[serde(default)]
    terminals: Vec<LiveTerminalEntry>,
}

#[derive(Deserialize)]
struct LiveTerminalEntry {
    session_id: u64,
    #[serde(default)]
    project_name: String,
}

fn terminals_path() -> Option<std::path::PathBuf> {
    let home = std::env::var_os("HOME")?;
    let mut p = std::path::PathBuf::from(home);
    p.push(".jim");
    p.push("terminals.json");
    Some(p)
}

fn load_terminals() -> Vec<LiveTerminalEntry> {
    let Some(p) = terminals_path() else { return vec![] };
    let Ok(bytes) = std::fs::read(&p) else {
        return vec![];
    };
    serde_json::from_slice::<LiveTerminals>(&bytes)
        .map(|t| t.terminals)
        .unwrap_or_default()
}

fn resolve_session(args: &Args) -> Result<u64, String> {
    if let Some(id) = args.session {
        return Ok(id);
    }
    if let Some(ref name) = args.project {
        let terms = load_terminals();
        let mut matched: Vec<&LiveTerminalEntry> = terms
            .iter()
            .filter(|t| t.project_name.eq_ignore_ascii_case(name))
            .collect();
        if matched.is_empty() {
            return Err(format!("no terminal found in project '{}'", name));
        }
        // Highest session_id is the most-recently spawned.
        matched.sort_by_key(|t| std::cmp::Reverse(t.session_id));
        return Ok(matched[0].session_id);
    }
    Err("specify --session <id> or --project <name>".into())
}

/// Parse a hex byte string like "1b0d" into [0x1b, 0x0d].
fn parse_hex_bytes(s: &str) -> Result<Vec<u8>, String> {
    let trimmed: String = s.chars().filter(|c| !c.is_whitespace()).collect();
    if trimmed.len() % 2 != 0 {
        return Err(format!("--bytes needs an even number of hex chars; got {}", trimmed.len()));
    }
    let mut out = Vec::with_capacity(trimmed.len() / 2);
    for chunk in trimmed.as_bytes().chunks(2) {
        let pair = std::str::from_utf8(chunk).map_err(|e| e.to_string())?;
        out.push(u8::from_str_radix(pair, 16).map_err(|e| format!("bad hex '{}': {}", pair, e))?);
    }
    Ok(out)
}

/// `\n`, `\r`, `\t`, `\\`, `\x1b`, etc. in --text args. We do this
/// ourselves so shells like `dash` that don't expand `$'\n'` still
/// work, and so the CLI is portable to scripted callers.
fn unescape(s: &str) -> Result<Vec<u8>, String> {
    let mut out: Vec<u8> = Vec::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '\\' {
            let mut buf = [0u8; 4];
            out.extend_from_slice(c.encode_utf8(&mut buf).as_bytes());
            continue;
        }
        let Some(n) = chars.next() else {
            return Err("trailing backslash".into());
        };
        let byte = match n {
            'n' => b'\n',
            'r' => b'\r',
            't' => b'\t',
            '\\' => b'\\',
            '0' => 0,
            'e' => 0x1b,
            'x' => {
                let h1 = chars.next().ok_or_else(|| "bad \\x escape".to_string())?;
                let h2 = chars.next().ok_or_else(|| "bad \\x escape".to_string())?;
                u8::from_str_radix(&format!("{}{}", h1, h2), 16)
                    .map_err(|e| format!("bad \\x escape: {}", e))?
            }
            other => return Err(format!("unknown escape \\{}", other)),
        };
        out.push(byte);
    }
    Ok(out)
}

struct Args {
    session: Option<u64>,
    project: Option<String>,
    payload: Vec<u8>,
}

impl Args {
    fn parse() -> Result<Self, String> {
        let mut session: Option<u64> = None;
        let mut project: Option<String> = None;
        let mut payload: Vec<u8> = Vec::new();

        let mut it = std::env::args().skip(1);
        while let Some(arg) = it.next() {
            match arg.as_str() {
                "-h" | "--help" => {
                    print_usage();
                    std::process::exit(0);
                }
                "--session" | "-s" => {
                    let v = it.next().ok_or_else(|| format!("{} needs a value", arg))?;
                    session = Some(v.parse().map_err(|e| format!("--session: {}", e))?);
                }
                "--project" | "-p" => {
                    project = Some(it.next().ok_or_else(|| format!("{} needs a value", arg))?);
                }
                "--text" | "-t" => {
                    let v = it.next().ok_or_else(|| format!("{} needs a value", arg))?;
                    payload.extend(unescape(&v)?);
                }
                "--bytes" | "-b" => {
                    let v = it.next().ok_or_else(|| format!("{} needs a value", arg))?;
                    payload.extend(parse_hex_bytes(&v)?);
                }
                "--stdin" => {
                    let mut buf = Vec::new();
                    std::io::stdin()
                        .read_to_end(&mut buf)
                        .map_err(|e| format!("read stdin: {}", e))?;
                    payload.extend(buf);
                }
                other => return Err(format!("unknown arg: {}", other)),
            }
        }
        if payload.is_empty() {
            return Err("nothing to inject — pass --text, --bytes, or --stdin".into());
        }
        Ok(Self {
            session,
            project,
            payload,
        })
    }
}

fn print_usage() {
    eprintln!(
        "tbinject (--session ID | --project NAME) [--text STR] [--bytes HEX] [--stdin]\n\
         \n\
         Inject bytes into a terminal-bevy session's PTY. Each --text\n\
         and --bytes flag appends to the payload; they may be repeated.\n\
         --text supports \\n \\r \\t \\\\ \\e (= ESC) \\xHH.\n\
         \n\
         Examples:\n\
           tbinject -s 27 -t 'hello\\n'\n\
           tbinject -p beagle -b 03           # Ctrl-C\n\
           tbinject -s 27 -b 1b               # Esc (cancel generation)\n\
           echo /compact | tbinject -s 27 --stdin\n"
    );
}

fn main() -> ExitCode {
    let args = match Args::parse() {
        Ok(a) => a,
        Err(e) => {
            eprintln!("tbinject: {}", e);
            print_usage();
            return ExitCode::from(2);
        }
    };

    let session = match resolve_session(&args) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("tbinject: {}", e);
            return ExitCode::from(1);
        }
    };

    let Some(sock) = terminal_daemon::inject_socket_path(session) else {
        eprintln!("tbinject: HOME unset, can't locate socket");
        return ExitCode::from(1);
    };

    let mut stream = match UnixStream::connect(&sock) {
        Ok(s) => s,
        Err(e) => {
            eprintln!(
                "tbinject: connect {}: {} (is the session daemon running?)",
                sock.display(),
                e
            );
            return ExitCode::from(1);
        }
    };

    if let Err(e) = stream.write_all(&args.payload) {
        eprintln!("tbinject: write: {}", e);
        return ExitCode::from(1);
    }
    // Half-close so the daemon sees EOF and stops reading. This is
    // what releases the connection in the daemon's inject loop.
    let _ = stream.shutdown(std::net::Shutdown::Write);
    ExitCode::SUCCESS
}
