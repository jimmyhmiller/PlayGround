//! Streaming OSC 7 (current-directory report) parser.
//!
//! OSC 7 is the de-facto-standard escape sequence shells emit to tell
//! the terminal "my cwd is now X":
//!
//!   ESC `]` `7` `;` `file://`<host>`/`<path> ST
//!
//! where ST (string terminator) is either BEL (`0x07`) or ESC `\`.
//!
//! Our shell-integration shim emits this on every prompt and on every
//! `cd`. The worker feeds the same bytes it hands to libghostty's
//! `vt_write` through [`Osc7Watcher`]; each completed sequence yields a
//! decoded path. Bytes may arrive split across `feed` calls (PTY reads
//! are arbitrary), so state is carried across calls.
//!
//! The implementation is deliberately tolerant: it only triggers on
//! the exact `ESC ] 7 ;` prefix; any other OSC is ignored. It also has
//! a hard size cap so a malformed sequence can't grow the buffer
//! without bound.

const MAX_PAYLOAD: usize = 4096;

#[derive(Debug)]
enum State {
    Normal,
    Esc,
    OscOpen,
    /// Inside an OSC 7 payload (after `ESC ] 7 ;`). Accumulating until ST.
    Osc7Payload,
    /// Inside an OSC we don't care about. Eat until ST, then drop.
    OscOther,
    /// Just saw ESC inside an OSC; if next byte is `\` we've hit ST.
    Osc7PayloadEsc,
    OscOtherEsc,
}

pub struct Osc7Watcher {
    state: State,
    buf: Vec<u8>,
}

impl Default for Osc7Watcher {
    fn default() -> Self {
        Self {
            state: State::Normal,
            buf: Vec::with_capacity(256),
        }
    }
}

impl Osc7Watcher {
    /// Feed a slice of PTY bytes through the watcher. For each
    /// completed OSC 7 sequence, `on_cwd` is invoked with the decoded
    /// filesystem path (URL host stripped, percent-decoding applied).
    /// Malformed payloads are silently dropped.
    pub fn feed<F: FnMut(String)>(&mut self, bytes: &[u8], mut on_cwd: F) {
        for &b in bytes {
            match self.state {
                State::Normal => {
                    if b == 0x1b {
                        self.state = State::Esc;
                    }
                }
                State::Esc => {
                    if b == b']' {
                        // OSC introducer. We don't yet know which OSC.
                        self.buf.clear();
                        self.state = State::OscOpen;
                    } else {
                        // Not an OSC after all (could be a CSI, SS3, etc.).
                        self.state = State::Normal;
                    }
                }
                State::OscOpen => {
                    if b == b';' {
                        // The prefix bytes before `;` identify which OSC.
                        if self.buf == b"7" {
                            self.buf.clear();
                            self.state = State::Osc7Payload;
                        } else {
                            self.buf.clear();
                            self.state = State::OscOther;
                        }
                    } else if b == 0x07 {
                        // BEL terminator before `;` — degenerate OSC, drop.
                        self.buf.clear();
                        self.state = State::Normal;
                    } else {
                        // Accumulate the OSC number (and any leading args).
                        if self.buf.len() < 8 {
                            self.buf.push(b);
                        } else {
                            // Number too long to be one we recognise.
                            self.state = State::OscOther;
                            self.buf.clear();
                        }
                    }
                }
                State::Osc7Payload => match b {
                    0x07 => {
                        self.emit(&mut on_cwd);
                        self.state = State::Normal;
                    }
                    0x1b => {
                        self.state = State::Osc7PayloadEsc;
                    }
                    _ => {
                        if self.buf.len() < MAX_PAYLOAD {
                            self.buf.push(b);
                        } else {
                            // Overflow — abandon this sequence; resync.
                            self.buf.clear();
                            self.state = State::OscOther;
                        }
                    }
                },
                State::Osc7PayloadEsc => {
                    if b == b'\\' {
                        // ESC \ — ST.
                        self.emit(&mut on_cwd);
                        self.state = State::Normal;
                    } else {
                        // Stray ESC inside payload — treat the ESC as a
                        // literal byte and keep going.
                        if self.buf.len() < MAX_PAYLOAD {
                            self.buf.push(0x1b);
                            self.buf.push(b);
                            self.state = State::Osc7Payload;
                        } else {
                            self.buf.clear();
                            self.state = State::OscOther;
                        }
                    }
                }
                State::OscOther => match b {
                    0x07 => {
                        self.state = State::Normal;
                    }
                    0x1b => {
                        self.state = State::OscOtherEsc;
                    }
                    _ => { /* eat */ }
                },
                State::OscOtherEsc => {
                    if b == b'\\' {
                        self.state = State::Normal;
                    } else {
                        // Stray ESC — keep eating as OSC.
                        self.state = State::OscOther;
                    }
                }
            }
        }
    }

    fn emit<F: FnMut(String)>(&mut self, on_cwd: &mut F) {
        let payload = std::mem::take(&mut self.buf);
        if let Some(path) = decode_file_url(&payload) {
            on_cwd(path);
        }
    }
}

/// Turn an OSC 7 payload into a local filesystem path. Accepts
/// `file://host/path`, `file:///path`, and bare paths. Drops the host
/// component (cwd reports may carry a hostname we don't care about
/// when the path is also valid locally). Returns None on payloads we
/// can't make sense of.
fn decode_file_url(payload: &[u8]) -> Option<String> {
    let s = std::str::from_utf8(payload).ok()?.trim();
    let path_str = if let Some(rest) = s.strip_prefix("file://") {
        // Split host/path on the first '/'. `file:///path` → host="", path="/path".
        match rest.find('/') {
            Some(i) => &rest[i..],
            None => return None,
        }
    } else if s.starts_with('/') {
        s
    } else {
        return None;
    };
    Some(percent_decode(path_str))
}

fn percent_decode(s: &str) -> String {
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'%' && i + 2 < bytes.len() {
            if let (Some(h), Some(l)) = (hex(bytes[i + 1]), hex(bytes[i + 2])) {
                out.push((h << 4) | l);
                i += 3;
                continue;
            }
        }
        out.push(bytes[i]);
        i += 1;
    }
    String::from_utf8_lossy(&out).into_owned()
}

fn hex(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(10 + b - b'a'),
        b'A'..=b'F' => Some(10 + b - b'A'),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn one(input: &[u8]) -> Vec<String> {
        let mut w = Osc7Watcher::default();
        let mut out = Vec::new();
        w.feed(input, |p| out.push(p));
        out
    }

    #[test]
    fn bel_terminator() {
        let s = b"\x1b]7;file://localhost/tmp/foo\x07";
        assert_eq!(one(s), vec!["/tmp/foo".to_string()]);
    }

    #[test]
    fn st_terminator() {
        let s = b"\x1b]7;file:///home/x\x1b\\";
        assert_eq!(one(s), vec!["/home/x".to_string()]);
    }

    #[test]
    fn split_across_feeds() {
        let mut w = Osc7Watcher::default();
        let mut out = Vec::new();
        w.feed(b"\x1b]7;file://", |p| out.push(p));
        w.feed(b"h/usr/local\x07tail", |p| out.push(p));
        assert_eq!(out, vec!["/usr/local".to_string()]);
    }

    #[test]
    fn ignores_other_oscs() {
        let s = b"\x1b]133;A\x07\x1b]7;file:///x\x07";
        assert_eq!(one(s), vec!["/x".to_string()]);
    }

    #[test]
    fn percent_decode_path() {
        let s = b"\x1b]7;file:///tmp/a%20b/c\x07";
        assert_eq!(one(s), vec!["/tmp/a b/c".to_string()]);
    }

    #[test]
    fn rejects_non_path() {
        let s = b"\x1b]7;not-a-url\x07";
        assert!(one(s).is_empty());
    }
}
