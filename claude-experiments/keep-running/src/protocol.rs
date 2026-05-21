use serde::{Deserialize, Serialize};

/// Messages sent from client to daemon
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ClientMessage {
    /// Attach to the session with terminal size
    Attach { cols: u16, rows: u16 },

    /// Input data from client's terminal
    Input(Vec<u8>),

    /// Terminal was resized
    Resize { cols: u16, rows: u16 },

    /// Client is detaching
    Detach,
}

/// Messages sent from daemon to client
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DaemonMessage {
    /// Successfully attached
    Attached,

    /// Output data from PTY
    Output(Vec<u8>),

    /// Replay of buffered history is starting (client should suppress rendering)
    ReplayStart,

    /// Replay of buffered history is complete (client should resume rendering)
    ReplayEnd,

    /// Child process exited
    ChildExited { code: Option<i32> },

    /// Error occurred
    Error(String),
}

/// Encode a message for wire transmission (length-prefixed bincode)
pub fn encode_message<T: Serialize>(msg: &T) -> anyhow::Result<Vec<u8>> {
    let payload = bincode::serde::encode_to_vec(msg, bincode::config::standard())?;
    let len = payload.len() as u32;
    let mut buf = len.to_be_bytes().to_vec();
    buf.extend(payload);
    Ok(buf)
}

/// Decode a length-prefixed message from a buffer
/// Returns (message, bytes_consumed) or None if incomplete
pub fn decode_message<T: for<'de> Deserialize<'de>>(buf: &[u8]) -> anyhow::Result<Option<(T, usize)>> {
    if buf.len() < 4 {
        return Ok(None);
    }

    let len = u32::from_be_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize;
    let total = 4 + len;

    if buf.len() < total {
        return Ok(None);
    }

    let (msg, _): (T, usize) =
        bincode::serde::decode_from_slice(&buf[4..total], bincode::config::standard())?;
    Ok(Some((msg, total)))
}
