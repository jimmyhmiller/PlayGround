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

    /// Child process exited
    ChildExited { code: Option<i32> },

    /// Error occurred
    Error(String),
}

/// Encode a message for wire transmission (length-prefixed JSON)
pub fn encode_message<T: Serialize>(msg: &T) -> anyhow::Result<Vec<u8>> {
    let json = serde_json::to_vec(msg)?;
    let len = json.len() as u32;
    let mut buf = len.to_be_bytes().to_vec();
    buf.extend(json);
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

    let msg: T = serde_json::from_slice(&buf[4..total])?;
    Ok(Some((msg, total)))
}
