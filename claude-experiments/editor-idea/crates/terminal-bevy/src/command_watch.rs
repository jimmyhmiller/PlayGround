//! Streaming OSC 133 (command mark) parser.
//!
//! Our shell-integration shim emits two marks per interactive command:
//!
//!   ESC `]` `133` `;` `C` `;` <base64-command> ST   (command starting)
//!   ESC `]` `133` `;` `D` `;` <exit-code>      ST   (command finished)
//!
//! where ST is BEL (`0x07`) or ESC `\`. The command line is base64'd by
//! the shim so newlines / ESC / `;` in the command can't corrupt the
//! OSC payload. We pair each `C` with the next `D` and, on `D`, invoke
//! the callback with the decoded command and its integer exit code.
//!
//! Structure mirrors [`crate::osc7::Osc7Watcher`]: a tolerant byte-fed
//! state machine that only triggers on the exact `ESC ] 133 ;` prefix
//! and ignores every other OSC. Bytes may split across `feed` calls.

const MAX_PAYLOAD: usize = 8192;

#[derive(Debug)]
enum State {
    Normal,
    Esc,
    OscOpen,
    /// Inside a 133 payload (after `ESC ] 133 ;`). Accumulates the
    /// `C;<b64>` / `D;<exit>` body until ST.
    Payload,
    /// Inside an OSC we don't care about. Eat until ST.
    Other,
    /// Saw ESC inside a 133 payload; `\` next ⇒ ST.
    PayloadEsc,
    OtherEsc,
}

pub struct CommandWatcher {
    state: State,
    buf: Vec<u8>,
    /// Command from the most recent `C` mark, awaiting its `D`.
    pending: Option<String>,
}

impl Default for CommandWatcher {
    fn default() -> Self {
        Self {
            state: State::Normal,
            buf: Vec::with_capacity(256),
            pending: None,
        }
    }
}

impl CommandWatcher {
    /// Feed PTY bytes. For each completed command (a `D` mark paired
    /// with the preceding `C`), `on_command(command, exit_code)` is
    /// invoked. Marks without a pending command, or malformed payloads,
    /// are silently dropped.
    pub fn feed<F: FnMut(String, i32)>(&mut self, bytes: &[u8], mut on_command: F) {
        for &b in bytes {
            match self.state {
                State::Normal => {
                    if b == 0x1b {
                        self.state = State::Esc;
                    }
                }
                State::Esc => {
                    if b == b']' {
                        self.buf.clear();
                        self.state = State::OscOpen;
                    } else {
                        self.state = State::Normal;
                    }
                }
                State::OscOpen => {
                    if b == b';' {
                        if self.buf == b"133" {
                            self.buf.clear();
                            self.state = State::Payload;
                        } else {
                            self.buf.clear();
                            self.state = State::Other;
                        }
                    } else if b == 0x07 {
                        self.buf.clear();
                        self.state = State::Normal;
                    } else if self.buf.len() < 8 {
                        self.buf.push(b);
                    } else {
                        self.state = State::Other;
                        self.buf.clear();
                    }
                }
                State::Payload => match b {
                    0x07 => {
                        self.emit(&mut on_command);
                        self.state = State::Normal;
                    }
                    0x1b => {
                        self.state = State::PayloadEsc;
                    }
                    _ => {
                        if self.buf.len() < MAX_PAYLOAD {
                            self.buf.push(b);
                        } else {
                            self.buf.clear();
                            self.state = State::Other;
                        }
                    }
                },
                State::PayloadEsc => {
                    if b == b'\\' {
                        self.emit(&mut on_command);
                        self.state = State::Normal;
                    } else if self.buf.len() + 2 <= MAX_PAYLOAD {
                        self.buf.push(0x1b);
                        self.buf.push(b);
                        self.state = State::Payload;
                    } else {
                        self.buf.clear();
                        self.state = State::Other;
                    }
                }
                State::Other => match b {
                    0x07 => self.state = State::Normal,
                    0x1b => self.state = State::OtherEsc,
                    _ => {}
                },
                State::OtherEsc => {
                    if b == b'\\' {
                        self.state = State::Normal;
                    } else {
                        self.state = State::Other;
                    }
                }
            }
        }
    }

    fn emit<F: FnMut(String, i32)>(&mut self, on_command: &mut F) {
        let payload = std::mem::take(&mut self.buf);
        let Ok(s) = std::str::from_utf8(&payload) else {
            return;
        };
        // Body is "C;<b64>" or "D;<exit>".
        let Some((kind, rest)) = s.split_once(';') else {
            return;
        };
        match kind {
            "C" => {
                if let Some(cmd) = base64_decode(rest).and_then(|b| String::from_utf8(b).ok()) {
                    let cmd = cmd.trim().to_string();
                    self.pending = (!cmd.is_empty()).then_some(cmd);
                }
            }
            "D" => {
                if let (Some(command), Ok(exit)) = (self.pending.take(), rest.trim().parse::<i32>())
                {
                    on_command(command, exit);
                }
            }
            _ => {}
        }
    }
}

/// Minimal standard-alphabet base64 decoder (no deps). Ignores
/// whitespace; returns None on invalid input.
fn base64_decode(s: &str) -> Option<Vec<u8>> {
    fn val(c: u8) -> Option<u8> {
        match c {
            b'A'..=b'Z' => Some(c - b'A'),
            b'a'..=b'z' => Some(c - b'a' + 26),
            b'0'..=b'9' => Some(c - b'0' + 52),
            b'+' => Some(62),
            b'/' => Some(63),
            _ => None,
        }
    }
    let mut out = Vec::with_capacity(s.len() / 4 * 3);
    let mut quad = [0u8; 4];
    let mut n = 0;
    for &c in s.as_bytes() {
        if c == b'=' || c.is_ascii_whitespace() {
            continue;
        }
        quad[n] = val(c)?;
        n += 1;
        if n == 4 {
            out.push((quad[0] << 2) | (quad[1] >> 4));
            out.push((quad[1] << 4) | (quad[2] >> 2));
            out.push((quad[2] << 6) | quad[3]);
            n = 0;
        }
    }
    match n {
        0 => {}
        2 => out.push((quad[0] << 2) | (quad[1] >> 4)),
        3 => {
            out.push((quad[0] << 2) | (quad[1] >> 4));
            out.push((quad[1] << 4) | (quad[2] >> 2));
        }
        _ => return None, // n == 1 is impossible for valid base64
    }
    Some(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(input: &[u8]) -> Vec<(String, i32)> {
        let mut w = CommandWatcher::default();
        let mut out = Vec::new();
        w.feed(input, |c, e| out.push((c, e)));
        out
    }

    // "cargo test" base64 = "Y2FyZ28gdGVzdA=="
    #[test]
    fn pairs_command_and_exit() {
        let s = b"\x1b]133;C;Y2FyZ28gdGVzdA==\x07\x1b]133;D;0\x07";
        assert_eq!(run(s), vec![("cargo test".to_string(), 0)]);
    }

    #[test]
    fn nonzero_exit() {
        let s = b"\x1b]133;C;Y2FyZ28gdGVzdA==\x1b\\\x1b]133;D;1\x1b\\";
        assert_eq!(run(s), vec![("cargo test".to_string(), 1)]);
    }

    #[test]
    fn d_without_c_is_dropped() {
        let s = b"\x1b]133;D;0\x07";
        assert!(run(s).is_empty());
    }

    #[test]
    fn ignores_other_oscs() {
        // OSC 7 cwd report interleaved should not confuse us.
        let s = b"\x1b]7;file:///x\x07\x1b]133;C;Y2FyZ28gdGVzdA==\x07\x1b]133;D;0\x07";
        assert_eq!(run(s), vec![("cargo test".to_string(), 0)]);
    }

    #[test]
    fn split_across_feeds() {
        let mut w = CommandWatcher::default();
        let mut out = Vec::new();
        w.feed(b"\x1b]133;C;Y2Fy", |c, e| out.push((c, e)));
        w.feed(b"Z28gdGVzdA==\x07\x1b]133;D;2\x07", |c, e| out.push((c, e)));
        assert_eq!(out, vec![("cargo test".to_string(), 2)]);
    }
}
