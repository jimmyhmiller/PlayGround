use byteorder::{BigEndian, ByteOrder};
use bytes::{BufMut, BytesMut};
use std::io::{Read, Write};
use std::net::TcpStream;

pub const MAGIC: u32 = 0xDA7A_1061;
// Bumped to 2 when token auth was added to the handshake. A v2 server and a
// v2 client always exchange an auth frame after the magic/version frame (the
// token is empty when auth is disabled), so the wire shape is uniform and a
// version-1 peer is cleanly rejected rather than left half-handshaked.
pub const VERSION: u32 = 2;

/// Largest auth token we will read at the handshake. Tokens are secrets, not
/// payloads — a few hundred bytes is plenty, and capping it stops an
/// unauthenticated peer from making us allocate.
const MAX_TOKEN_LEN: u32 = 4096;

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
    #[error("Token too large: {0} bytes")]
    TokenTooLarge(u32),
    #[error("Authentication failed: {0}")]
    AuthFailed(String),
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
///
/// `expected_token` is the server's configured shared secret. When it is
/// `None`, auth is disabled and the presented token is ignored (but still
/// read off the wire, because a v2 client always sends the frame). When it
/// is `Some`, the client's token must match it byte-for-byte — compared in
/// constant time so a network attacker can't time the comparison to recover
/// the secret one byte at a time.
pub fn server_handshake(stream: &mut TcpStream, expected_token: Option<&[u8]>) -> Result<()> {
    let mut buf = [0u8; 8];
    stream.read_exact(&mut buf)?;

    let magic = BigEndian::read_u32(&buf[0..4]);
    let version = BigEndian::read_u32(&buf[4..8]);

    if magic != MAGIC {
        let err_msg = "invalid magic number";
        send_handshake_error(stream, err_msg)?;
        return Err(ProtocolError::InvalidMagic);
    }

    if version != VERSION {
        let err_msg = format!("unsupported version: {}", version);
        send_handshake_error(stream, &err_msg)?;
        return Err(ProtocolError::UnsupportedVersion(version));
    }

    // Read the auth frame: a 4-byte big-endian length followed by the token
    // bytes. Always present in v2, even when the token is empty.
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf)?;
    let token_len = BigEndian::read_u32(&len_buf);
    if token_len > MAX_TOKEN_LEN {
        send_handshake_error(stream, "token too large")?;
        return Err(ProtocolError::TokenTooLarge(token_len));
    }
    let mut token = vec![0u8; token_len as usize];
    stream.read_exact(&mut token)?;

    if let Some(expected) = expected_token {
        if !constant_time_eq(&token, expected) {
            // Deliberately generic message — don't reveal whether it was an
            // empty token, wrong length, etc.
            send_handshake_error(stream, "authentication failed")?;
            return Err(ProtocolError::AuthFailed("invalid token".into()));
        }
    }

    // Send OK
    stream.write_all(&[0x00])?;
    stream.flush()?;
    Ok(())
}

/// Compare two byte slices in time independent of their contents.
///
/// The comparison time depends only on `a.len()`, never on where (or whether)
/// the bytes differ. Returns false immediately on a length mismatch — the
/// length of a secret is not itself secret here, and folding differing
/// lengths into the loop would compare against out-of-range indices.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff: u8 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

fn send_handshake_error(stream: &mut TcpStream, msg: &str) -> Result<()> {
    let msg_bytes = msg.as_bytes();
    let mut buf = vec![0x01];
    buf.extend_from_slice(&(msg_bytes.len() as u32).to_be_bytes());
    buf.extend_from_slice(msg_bytes);
    stream.write_all(&buf)?;
    stream.flush()?;
    Ok(())
}

/// Read a framed message from the stream.
pub fn read_message(stream: &mut TcpStream) -> Result<Message> {
    // Read header: request_id (8) + payload_length (4)
    let mut header = [0u8; 12];
    stream.read_exact(&mut header)?;

    let request_id = BigEndian::read_u64(&header[0..8]);
    let payload_length = BigEndian::read_u32(&header[8..12]);

    if payload_length > MAX_PAYLOAD_SIZE {
        return Err(ProtocolError::PayloadTooLarge(payload_length));
    }

    let mut payload_buf = vec![0u8; payload_length as usize];
    stream.read_exact(&mut payload_buf)?;

    let payload: serde_json::Value = serde_json::from_slice(&payload_buf)
        .map_err(|e| ProtocolError::InvalidJson(e.to_string()))?;

    Ok(Message {
        request_id,
        payload,
    })
}

/// Write a framed response message to the stream.
pub fn write_message(stream: &mut TcpStream, request_id: u64, payload: &serde_json::Value) -> Result<()> {
    let payload_bytes = serde_json::to_vec(payload)
        .map_err(|e| ProtocolError::InvalidJson(e.to_string()))?;

    let mut buf = BytesMut::with_capacity(12 + payload_bytes.len());
    buf.put_u64(request_id);
    buf.put_u32(payload_bytes.len() as u32);
    buf.put_slice(&payload_bytes);

    stream.write_all(&buf)?;
    stream.flush()?;
    Ok(())
}

/// Perform client-side handshake, presenting `token` for authentication.
///
/// Pass an empty slice when connecting to a server that has auth disabled —
/// the frame is still sent so the wire shape matches the server's read.
pub fn client_handshake(stream: &mut TcpStream, token: &[u8]) -> Result<()> {
    let mut buf = [0u8; 8];
    BigEndian::write_u32(&mut buf[0..4], MAGIC);
    BigEndian::write_u32(&mut buf[4..8], VERSION);
    stream.write_all(&buf)?;

    // Auth frame: 4-byte length + token bytes.
    let mut auth = Vec::with_capacity(4 + token.len());
    auth.extend_from_slice(&(token.len() as u32).to_be_bytes());
    auth.extend_from_slice(token);
    stream.write_all(&auth)?;
    stream.flush()?;

    // Read response
    let mut resp = [0u8; 1];
    stream.read_exact(&mut resp)?;

    if resp[0] == 0x00 {
        Ok(())
    } else {
        // Read error message
        let mut len_buf = [0u8; 4];
        stream.read_exact(&mut len_buf)?;
        let len = BigEndian::read_u32(&len_buf) as usize;
        let mut msg_buf = vec![0u8; len];
        stream.read_exact(&mut msg_buf)?;
        let msg = String::from_utf8_lossy(&msg_buf).to_string();
        // The server sends "authentication failed" for a bad/missing token;
        // surface that as AuthFailed so callers can distinguish it from a
        // generic protocol problem. Other handshake errors stay InvalidJson.
        if msg == "authentication failed" {
            Err(ProtocolError::AuthFailed(msg))
        } else {
            Err(ProtocolError::InvalidJson(msg))
        }
    }
}
