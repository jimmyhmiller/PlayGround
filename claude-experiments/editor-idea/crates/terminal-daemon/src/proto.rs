//! Wire protocol between the editor (client) and a per-terminal daemon.
//!
//! Length-prefixed bincode frames over a Unix stream socket. The protocol
//! is intentionally internal to this crate — the daemon and client are
//! built from the same source tree and we can evolve the format without
//! worrying about external consumers.
//!
//! Wire format: `[u32 big-endian length][bincode payload]`.

use serde::{Deserialize, Serialize};

/// Messages sent from client to daemon.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClientMessage {
    /// First message after connecting. Resizes the PTY to the client's
    /// current cell grid and triggers a replay of the daemon's history
    /// buffer.
    Attach { cols: u16, rows: u16 },
    /// Bytes to write to the PTY (keystrokes, paste blobs, VT responses).
    Input(Vec<u8>),
    /// PTY winsize change.
    Resize { cols: u16, rows: u16 },
    /// Client is going away gracefully. Daemon keeps running.
    Detach,
    /// Stop the daemon. The child is killed and the session is torn down.
    Kill,
}

/// Messages sent from daemon to client.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DaemonMessage {
    /// Acknowledgement of Attach. Replay (if any) begins after this.
    Attached,
    /// Bytes the PTY produced.
    Output(Vec<u8>),
    /// Replay of buffered history is starting. Client should suppress
    /// snapshot publishes until ReplayEnd so the renderer doesn't flicker
    /// through dozens of intermediate states.
    ReplayStart,
    /// Replay complete. Subsequent Output is live.
    ReplayEnd,
    /// Child process exited.
    ChildExited { code: Option<i32> },
}

/// Length-prefix overhead.
const LEN_PREFIX: usize = 4;

fn bincode_config() -> bincode::config::Configuration {
    bincode::config::standard()
}

/// Encode a message into a length-prefixed frame.
pub fn encode<T: Serialize>(msg: &T) -> Result<Vec<u8>, bincode::error::EncodeError> {
    let payload = bincode::serde::encode_to_vec(msg, bincode_config())?;
    let len = payload.len() as u32;
    let mut buf = Vec::with_capacity(LEN_PREFIX + payload.len());
    buf.extend_from_slice(&len.to_be_bytes());
    buf.extend_from_slice(&payload);
    Ok(buf)
}

/// Decode one message from the head of `buf`.
///
/// Returns `Ok(Some((msg, consumed)))` on a complete frame, `Ok(None)`
/// when more bytes are needed, and `Err` on a corrupt frame (caller
/// should drop the connection).
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
    let (msg, _) = bincode::serde::decode_from_slice(&buf[LEN_PREFIX..total], bincode_config())?;
    Ok(Some((msg, total)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_client_msg() {
        let m = ClientMessage::Input(vec![1, 2, 3, 4, 5]);
        let bytes = encode(&m).unwrap();
        let (decoded, n): (ClientMessage, _) = decode(&bytes).unwrap().unwrap();
        assert_eq!(n, bytes.len());
        match decoded {
            ClientMessage::Input(v) => assert_eq!(v, vec![1, 2, 3, 4, 5]),
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn decode_returns_none_on_partial() {
        let bytes = encode(&ClientMessage::Attach { cols: 80, rows: 24 }).unwrap();
        let truncated = &bytes[..bytes.len() - 1];
        let res: Result<Option<(ClientMessage, _)>, _> = decode(truncated);
        assert!(matches!(res, Ok(None)));
    }

    #[test]
    fn decode_two_messages_in_a_row() {
        let mut buf = encode(&ClientMessage::Attach { cols: 80, rows: 24 }).unwrap();
        buf.extend_from_slice(&encode(&ClientMessage::Input(vec![b'x'])).unwrap());

        let (m1, n1): (ClientMessage, _) = decode(&buf).unwrap().unwrap();
        assert!(matches!(m1, ClientMessage::Attach { cols: 80, rows: 24 }));
        let (m2, n2): (ClientMessage, _) = decode(&buf[n1..]).unwrap().unwrap();
        assert!(matches!(m2, ClientMessage::Input(ref v) if v == &vec![b'x']));
        assert_eq!(n1 + n2, buf.len());
    }
}
