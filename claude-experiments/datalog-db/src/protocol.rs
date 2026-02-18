use byteorder::{BigEndian, ByteOrder};
use bytes::{BufMut, BytesMut};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpStream;

pub const MAGIC: u32 = 0xDA7A_1061;
pub const VERSION: u32 = 1;

#[derive(Debug, thiserror::Error)]
pub enum ProtocolError {
    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
    #[error("Invalid magic number")]
    InvalidMagic,
    #[error("Unsupported version: {0}")]
    UnsupportedVersion(u32),
    #[error("Payload too large: {0} bytes")]
    PayloadTooLarge(u32),
    #[error("Invalid JSON: {0}")]
    InvalidJson(String),
}

pub type Result<T> = std::result::Result<T, ProtocolError>;

const MAX_PAYLOAD_SIZE: u32 = 64 * 1024 * 1024; // 64 MB

/// A framed message with a request ID and JSON payload.
#[derive(Debug)]
pub struct Message {
    pub request_id: u64,
    pub payload: serde_json::Value,
}

/// Perform server-side handshake.
pub async fn server_handshake(stream: &mut TcpStream) -> Result<()> {
    let mut buf = [0u8; 8];
    stream.read_exact(&mut buf).await?;

    let magic = BigEndian::read_u32(&buf[0..4]);
    let version = BigEndian::read_u32(&buf[4..8]);

    if magic != MAGIC {
        let err_msg = "invalid magic number";
        send_handshake_error(stream, err_msg).await?;
        return Err(ProtocolError::InvalidMagic);
    }

    if version != VERSION {
        let err_msg = format!("unsupported version: {}", version);
        send_handshake_error(stream, &err_msg).await?;
        return Err(ProtocolError::UnsupportedVersion(version));
    }

    // Send OK
    stream.write_all(&[0x00]).await?;
    stream.flush().await?;
    Ok(())
}

async fn send_handshake_error(stream: &mut TcpStream, msg: &str) -> Result<()> {
    let msg_bytes = msg.as_bytes();
    let mut buf = vec![0x01];
    buf.extend_from_slice(&(msg_bytes.len() as u32).to_be_bytes());
    buf.extend_from_slice(msg_bytes);
    stream.write_all(&buf).await?;
    stream.flush().await?;
    Ok(())
}

/// Read a framed message from the stream.
pub async fn read_message(stream: &mut TcpStream) -> Result<Message> {
    // Read header: request_id (8) + payload_length (4)
    let mut header = [0u8; 12];
    stream.read_exact(&mut header).await?;

    let request_id = BigEndian::read_u64(&header[0..8]);
    let payload_length = BigEndian::read_u32(&header[8..12]);

    if payload_length > MAX_PAYLOAD_SIZE {
        return Err(ProtocolError::PayloadTooLarge(payload_length));
    }

    let mut payload_buf = vec![0u8; payload_length as usize];
    stream.read_exact(&mut payload_buf).await?;

    let payload: serde_json::Value = serde_json::from_slice(&payload_buf)
        .map_err(|e| ProtocolError::InvalidJson(e.to_string()))?;

    Ok(Message {
        request_id,
        payload,
    })
}

/// Write a framed response message to the stream.
pub async fn write_message(stream: &mut TcpStream, request_id: u64, payload: &serde_json::Value) -> Result<()> {
    let payload_bytes = serde_json::to_vec(payload)
        .map_err(|e| ProtocolError::InvalidJson(e.to_string()))?;

    let mut buf = BytesMut::with_capacity(12 + payload_bytes.len());
    buf.put_u64(request_id);
    buf.put_u32(payload_bytes.len() as u32);
    buf.put_slice(&payload_bytes);

    stream.write_all(&buf).await?;
    stream.flush().await?;
    Ok(())
}

/// Perform client-side handshake.
pub async fn client_handshake(stream: &mut TcpStream) -> Result<()> {
    let mut buf = [0u8; 8];
    BigEndian::write_u32(&mut buf[0..4], MAGIC);
    BigEndian::write_u32(&mut buf[4..8], VERSION);
    stream.write_all(&buf).await?;
    stream.flush().await?;

    // Read response
    let mut resp = [0u8; 1];
    stream.read_exact(&mut resp).await?;

    if resp[0] == 0x00 {
        Ok(())
    } else {
        // Read error message
        let mut len_buf = [0u8; 4];
        stream.read_exact(&mut len_buf).await?;
        let len = BigEndian::read_u32(&len_buf) as usize;
        let mut msg_buf = vec![0u8; len];
        stream.read_exact(&mut msg_buf).await?;
        let msg = String::from_utf8_lossy(&msg_buf).to_string();
        Err(ProtocolError::InvalidJson(msg))
    }
}
