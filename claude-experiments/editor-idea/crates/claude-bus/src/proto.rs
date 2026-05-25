//! Wire protocol for `claude-bus`.
//!
//! Length-prefixed bincode frames: `[u32 BE length][bincode payload]`.
//! Same shape as `terminal-daemon::proto` so the rest of the codebase
//! sees one consistent IPC style.
//!
//! Two roles share the socket:
//!
//! * **Publisher** — opens the socket, writes one `ClientFrame::Hello {
//!   role: Publisher }` followed by one-or-more `ClientFrame::Publish`
//!   frames, then closes. Fire-and-forget; the daemon never replies on
//!   the publisher channel. This keeps hook latency at a single
//!   `connect()` + `write()`.
//!
//! * **Subscriber** — opens the socket, sends `ClientFrame::Hello {
//!   role: Subscriber { since_seq } }`, and stays connected. The daemon
//!   replays from `since_seq` (clamped to what the in-memory ring still
//!   has) and then streams live `BusFrame::Event` frames.

use serde::{Deserialize, Serialize};

const LEN_PREFIX: usize = 4;

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum Role {
    /// Will only send `Publish` frames, never reads from the socket.
    Publisher,
    /// Wants to receive events. `since_seq = None` means "live only";
    /// `Some(n)` means "everything with seq >= n that we still have,
    /// then live". Subscribers that miss the ring window get a
    /// `BusFrame::Lagged` marker so they can fall back to the JSONL
    /// file from their last known seq.
    Subscriber { since_seq: Option<u64> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClientFrame {
    /// MUST be the first frame on every connection. Identifies the
    /// caller's role.
    Hello { role: Role },
    /// Publisher → daemon. One event. The daemon assigns a sequence
    /// number, appends to JSONL, and broadcasts.
    ///
    /// `payload_json` is the hook's own JSON object as a string — we
    /// keep it opaque here so adding new hook payload fields doesn't
    /// require a protocol bump.
    Publish {
        kind: String,
        ts: u64,
        terminal_session_id: String,
        claude_pid: u32,
        payload_json: String,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BusFrame {
    /// Daemon → subscriber. One delivered event.
    Event {
        seq: u64,
        kind: String,
        ts: u64,
        terminal_session_id: String,
        claude_pid: u32,
        payload_json: String,
    },
    /// Daemon → subscriber. Sent at the start of a `Subscribe` with
    /// `since_seq` smaller than the oldest seq the ring still holds.
    /// The subscriber should consult the JSONL file from `requested`
    /// up to (but not including) `replay_from`, then continue with the
    /// `Event` frames that follow.
    Lagged { requested: u64, replay_from: u64 },
    /// Daemon → subscriber. Replay segment finished; everything after
    /// is live. Optional — subscribers can also just treat all
    /// `Event` frames uniformly.
    ReplayEnd,
}

pub fn encode<T: Serialize>(msg: &T) -> Result<Vec<u8>, bincode::error::EncodeError> {
    let payload = bincode::serde::encode_to_vec(msg, bincode::config::standard())?;
    let len = payload.len() as u32;
    let mut buf = Vec::with_capacity(LEN_PREFIX + payload.len());
    buf.extend_from_slice(&len.to_be_bytes());
    buf.extend_from_slice(&payload);
    Ok(buf)
}

/// Decode one frame from the head of `buf`. Returns `Ok(None)` if more
/// bytes are needed; `Err` only on a malformed payload (caller should
/// drop the connection).
pub fn decode<T: for<'de> Deserialize<'de>>(
    buf: &[u8],
) -> Result<Option<(T, usize)>, bincode::error::DecodeError> {
    if buf.len() < LEN_PREFIX {
        return Ok(None);
    }
    let len = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
    let total = LEN_PREFIX + len;
    if buf.len() < total {
        return Ok(None);
    }
    let (msg, _) = bincode::serde::decode_from_slice(&buf[LEN_PREFIX..total], bincode::config::standard())?;
    Ok(Some((msg, total)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_publish() {
        let m = ClientFrame::Publish {
            kind: "pre_tool_use".into(),
            ts: 1779294304,
            terminal_session_id: "31".into(),
            claude_pid: 80618,
            payload_json: r#"{"tool":"Bash"}"#.into(),
        };
        let bytes = encode(&m).unwrap();
        let (decoded, n): (ClientFrame, _) = decode(&bytes).unwrap().unwrap();
        assert_eq!(n, bytes.len());
        match decoded {
            ClientFrame::Publish { kind, ts, .. } => {
                assert_eq!(kind, "pre_tool_use");
                assert_eq!(ts, 1779294304);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn decode_partial_is_none() {
        let bytes = encode(&ClientFrame::Hello { role: Role::Publisher }).unwrap();
        let res: Result<Option<(ClientFrame, _)>, _> = decode(&bytes[..bytes.len() - 1]);
        assert!(matches!(res, Ok(None)));
    }
}
