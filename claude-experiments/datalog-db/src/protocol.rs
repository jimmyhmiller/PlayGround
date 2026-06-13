//! datalog-db wire protocol: framing, handshake, and authentication.
//!
//! A connection begins with a fixed handshake — 8 bytes of magic+version, then
//! a **method-negotiated auth exchange** — after which the message loop sends
//! length-prefixed JSON request/response frames.
//!
//! ## Transport independence
//!
//! Every function here is generic over `Read`/`Write`, so the exact same code
//! runs over a plain [`std::net::TcpStream`] or over a rustls TLS stream. TLS,
//! when enabled, is a *wrapper* established before byte zero (see
//! [`crate::transport`]); the handshake/framing below is byte-identical inside
//! the tunnel.
//!
//! ## Auth methods (VERSION 3)
//!
//! After magic+version the client sends a small JSON control frame naming the
//! method, and the exchange proceeds accordingly:
//!
//! - `{"method":"token"}` — legacy shared-bearer token. The client then sends a
//!   token frame; the server constant-time compares it. This is the path the
//!   loopback prod consumers use, unchanged from VERSION 2's semantics.
//! - `{"method":"scram","user":"<name>"}` — per-user SCRAM-SHA-256 (RFC 5802 /
//!   7677). The four SASL messages follow, each in its own frame. Confidentiality
//!   is expected to come from an outer TLS tunnel.
//!
//! Both paths end with the server's terminal status byte (`0x00` OK,
//! `0x01`+len+utf8 error) so the message loop starts from a known point.

use byteorder::{BigEndian, ByteOrder};
use bytes::{BufMut, BytesMut};
use std::io::{Read, Write};

use crate::auth::scram::{self, ScramServer, Verifier};

pub const MAGIC: u32 = 0xDA7A_1061;
// VERSION 3 replaced the fixed token frame with a method-negotiated auth
// exchange (token | scram). A v3 server still speaks the token sub-protocol, so
// VERSION-2-era token clients are migrated by re-tagging, but a peer that sends
// the literal integer 2 is cleanly rejected rather than left half-handshaked.
pub const VERSION: u32 = 3;

/// Largest single auth frame we will read at the handshake (token bytes, a
/// SASL message, or the JSON method header). Secrets and SASL messages are
/// small; capping this stops an unauthenticated peer from forcing a large
/// allocation before it has proven anything.
const MAX_AUTH_FRAME: u32 = 8192;

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
    #[error("Auth frame too large: {0} bytes")]
    AuthFrameTooLarge(u32),
    #[error("Authentication failed: {0}")]
    AuthFailed(String),
    #[error("Unsupported auth method: {0}")]
    UnsupportedAuthMethod(String),
    #[error("SCRAM error: {0}")]
    Scram(String),
}

impl From<scram::ScramError> for ProtocolError {
    fn from(e: scram::ScramError) -> Self {
        ProtocolError::Scram(e.to_string())
    }
}

pub type Result<T> = std::result::Result<T, ProtocolError>;

const MAX_PAYLOAD_SIZE: u32 = 64 * 1024 * 1024; // 64 MB

/// A framed message with a request ID and JSON payload.
#[derive(Debug)]
pub struct Message {
    pub request_id: u64,
    pub payload: serde_json::Value,
}

/// What the server is willing to accept, and where to look up SCRAM users.
///
/// The lifetime'd trait object keeps `protocol` from depending on `db` — the
/// server passes an adapter over [`crate::db::Database`].
pub struct ServerAuth<'a> {
    /// Shared bearer token. `Some` enables the `token` method; `None` disables
    /// it (a `token` attempt is rejected).
    pub expected_token: Option<Vec<u8>>,
    /// User store for the `scram` method. `Some` enables SCRAM.
    pub users: Option<&'a dyn UserStore>,
    /// Server-only key used to derive a deterministic mock verifier for unknown
    /// users, so a failed SCRAM login can't be distinguished from a wrong
    /// password (enumeration resistance).
    pub mock_key: &'a [u8],
    /// When true, the server runs open: the `token` method accepts any token
    /// (including empty). This is the explicit `--no-auth` mode. Without it, a
    /// `token` attempt against a server with no configured token is rejected.
    pub allow_no_auth: bool,
}

/// Abstraction over the SCRAM verifier store so `protocol` need not know about
/// the database. Implemented for `Database` in `server.rs`.
pub trait UserStore: Send + Sync {
    /// Look up a user's verifier. `Ok(None)` = no such user (the caller will
    /// mock-authenticate); `Err` = a backend/corruption failure.
    fn lookup(&self, username: &str) -> std::result::Result<Option<Verifier>, String>;
}

/// The identity established by a successful handshake. `user` is `Some` for a
/// SCRAM login, `None` for token auth or no-auth.
#[derive(Debug, Clone, Default)]
pub struct AuthOutcome {
    pub user: Option<String>,
}

// ---------------------------------------------------------------------------
// Low-level length-prefixed frame helpers (4-byte BE length + bytes)
// ---------------------------------------------------------------------------

/// Read one `len(u32-BE) + bytes` frame, capped at `MAX_AUTH_FRAME`.
fn read_frame<S: Read>(stream: &mut S) -> Result<Vec<u8>> {
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf)?;
    let len = BigEndian::read_u32(&len_buf);
    if len > MAX_AUTH_FRAME {
        return Err(ProtocolError::AuthFrameTooLarge(len));
    }
    let mut buf = vec![0u8; len as usize];
    stream.read_exact(&mut buf)?;
    Ok(buf)
}

/// Write one `len(u32-BE) + bytes` frame and flush.
fn write_frame<S: Write>(stream: &mut S, bytes: &[u8]) -> Result<()> {
    let mut out = Vec::with_capacity(4 + bytes.len());
    out.extend_from_slice(&(bytes.len() as u32).to_be_bytes());
    out.extend_from_slice(bytes);
    stream.write_all(&out)?;
    stream.flush()?;
    Ok(())
}

/// Compare two byte slices in time independent of their contents. Returns false
/// immediately on a length mismatch (a secret's length is not itself secret).
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    use subtle::ConstantTimeEq;
    if a.len() != b.len() {
        return false;
    }
    a.ct_eq(b).into()
}

fn send_ok<S: Write>(stream: &mut S) -> Result<()> {
    stream.write_all(&[0x00])?;
    stream.flush()?;
    Ok(())
}

fn send_handshake_error<S: Write>(stream: &mut S, msg: &str) -> Result<()> {
    let msg_bytes = msg.as_bytes();
    let mut buf = vec![0x01];
    buf.extend_from_slice(&(msg_bytes.len() as u32).to_be_bytes());
    buf.extend_from_slice(msg_bytes);
    stream.write_all(&buf)?;
    stream.flush()?;
    Ok(())
}

// ---------------------------------------------------------------------------
// Server handshake
// ---------------------------------------------------------------------------

/// Perform the server-side handshake and authenticate the client.
///
/// Reads magic+version, then the method-tagged auth exchange. On success the
/// server has sent its terminal `0x00` byte and the connection is ready for the
/// message loop; the returned [`AuthOutcome`] names the authenticated user (if
/// SCRAM). On failure the server has sent a `0x01` error frame and an `Err` is
/// returned so the caller closes the connection.
pub fn server_handshake<S: Read + Write>(
    stream: &mut S,
    auth: &ServerAuth,
) -> Result<AuthOutcome> {
    let mut buf = [0u8; 8];
    stream.read_exact(&mut buf)?;
    let magic = BigEndian::read_u32(&buf[0..4]);
    let version = BigEndian::read_u32(&buf[4..8]);

    if magic != MAGIC {
        send_handshake_error(stream, "invalid magic number")?;
        return Err(ProtocolError::InvalidMagic);
    }
    if version != VERSION {
        send_handshake_error(stream, &format!("unsupported version: {version}"))?;
        return Err(ProtocolError::UnsupportedVersion(version));
    }

    // Method control frame: small JSON { "method": "...", ... }.
    let header_bytes = read_frame(stream)?;
    let header: serde_json::Value = serde_json::from_slice(&header_bytes)
        .map_err(|e| ProtocolError::InvalidJson(e.to_string()))?;
    let method = header.get("method").and_then(|m| m.as_str()).unwrap_or("");

    match method {
        "token" => {
            server_token_auth(stream, auth)?;
            Ok(AuthOutcome { user: None })
        }
        "scram" => {
            let user = header
                .get("user")
                .and_then(|u| u.as_str())
                .unwrap_or("")
                .to_string();
            let outcome = server_scram_auth(stream, auth, &user)?;
            Ok(outcome)
        }
        other => {
            send_handshake_error(stream, &format!("unsupported auth method: {other}"))?;
            Err(ProtocolError::UnsupportedAuthMethod(other.to_string()))
        }
    }
}

fn server_token_auth<S: Read + Write>(stream: &mut S, auth: &ServerAuth) -> Result<()> {
    let token = read_frame(stream)?;
    let expected = match &auth.expected_token {
        Some(t) => t,
        None => {
            // No token configured. In explicit --no-auth mode the server is
            // open and accepts any token; otherwise a token attempt is rejected.
            if auth.allow_no_auth {
                return send_ok(stream);
            }
            send_handshake_error(stream, "authentication failed")?;
            return Err(ProtocolError::AuthFailed("token auth not enabled".into()));
        }
    };
    if constant_time_eq(&token, expected) {
        send_ok(stream)
    } else {
        send_handshake_error(stream, "authentication failed")?;
        Err(ProtocolError::AuthFailed("invalid token".into()))
    }
}

fn server_scram_auth<S: Read + Write>(
    stream: &mut S,
    auth: &ServerAuth,
    username: &str,
) -> Result<AuthOutcome> {
    let users = match auth.users {
        Some(u) => u,
        None => {
            send_handshake_error(stream, "authentication failed")?;
            return Err(ProtocolError::AuthFailed("scram auth not enabled".into()));
        }
    };

    // Build a real or a mock SCRAM server depending on whether the user exists.
    // The mock path runs the full exchange so timing/shape don't reveal which.
    let mut scram = match users.lookup(username) {
        Ok(Some(verifier)) => ScramServer::new(verifier),
        Ok(None) => ScramServer::mock(username, auth.mock_key),
        Err(e) => {
            send_handshake_error(stream, "authentication failed")?;
            return Err(ProtocolError::Scram(format!("user lookup failed: {e}")));
        }
    };

    // 1. client-first
    let client_first = read_string_frame(stream)?;
    // 2. server-first
    let server_first = scram.server_first(&client_first)?;
    write_frame(stream, server_first.as_bytes())?;
    // 3. client-final
    let client_final = read_string_frame(stream)?;
    // 4. server-final (+ ok flag)
    let (server_final, ok) = scram.server_final(&client_final)?;
    write_frame(stream, server_final.as_bytes())?;

    if ok {
        send_ok(stream)?;
        Ok(AuthOutcome {
            user: Some(username.to_string()),
        })
    } else {
        send_handshake_error(stream, "authentication failed")?;
        Err(ProtocolError::AuthFailed("invalid credentials".into()))
    }
}

fn read_string_frame<S: Read>(stream: &mut S) -> Result<String> {
    let bytes = read_frame(stream)?;
    String::from_utf8(bytes).map_err(|_| ProtocolError::Scram("non-utf8 SCRAM message".into()))
}

// ---------------------------------------------------------------------------
// Client handshake
// ---------------------------------------------------------------------------

fn write_magic_version<S: Write>(stream: &mut S) -> Result<()> {
    let mut buf = [0u8; 8];
    BigEndian::write_u32(&mut buf[0..4], MAGIC);
    BigEndian::write_u32(&mut buf[4..8], VERSION);
    stream.write_all(&buf)?;
    Ok(())
}

/// Read the server's terminal handshake status byte. `0x00` = OK; `0x01`
/// precedes a `len+utf8` error message, surfaced as `AuthFailed` for the
/// generic "authentication failed" text and `InvalidJson` otherwise.
fn read_status<S: Read>(stream: &mut S) -> Result<()> {
    let mut resp = [0u8; 1];
    stream.read_exact(&mut resp)?;
    if resp[0] == 0x00 {
        return Ok(());
    }
    let mut len_buf = [0u8; 4];
    stream.read_exact(&mut len_buf)?;
    let len = BigEndian::read_u32(&len_buf) as usize;
    let mut msg_buf = vec![0u8; len];
    stream.read_exact(&mut msg_buf)?;
    let msg = String::from_utf8_lossy(&msg_buf).to_string();
    if msg == "authentication failed" {
        Err(ProtocolError::AuthFailed(msg))
    } else {
        Err(ProtocolError::InvalidJson(msg))
    }
}

/// Client-side handshake using the shared bearer `token` method.
///
/// Pass an empty slice for a `--no-auth` server (the frame is still sent so the
/// wire shape matches the server's read).
pub fn client_handshake<S: Read + Write>(stream: &mut S, token: &[u8]) -> Result<()> {
    write_magic_version(stream)?;
    write_frame(stream, br#"{"method":"token"}"#)?;
    write_frame(stream, token)?;
    stream.flush()?;
    read_status(stream)
}

/// Client-side handshake using per-user SCRAM-SHA-256.
///
/// On success the client has verified the server's `ServerSignature` (mutual
/// auth) before returning `Ok`. A wrong password, unknown user, or impostor
/// server all surface as an error.
pub fn client_handshake_scram<S: Read + Write>(
    stream: &mut S,
    username: &str,
    password: &str,
) -> Result<()> {
    write_magic_version(stream)?;
    let header = serde_json::json!({ "method": "scram", "user": username });
    write_frame(stream, serde_json::to_vec(&header).unwrap().as_slice())?;

    let mut client = scram::ScramClient::new(username, password);
    // 1. client-first
    let client_first = client.client_first();
    write_frame(stream, client_first.as_bytes())?;
    // 2. server-first
    let server_first = read_string_frame(stream)?;
    // 3. client-final
    let client_final = client.client_final(&server_first)?;
    write_frame(stream, client_final.as_bytes())?;
    // 4. server-final — verify the server signature BEFORE trusting the status.
    //    A failure here means either a wrong password (the keys we derived don't
    //    match the verifier, so the server's honest signature won't verify) or
    //    an impostor server that doesn't know ServerKey. The client cannot
    //    distinguish the two and must trust neither, so both map to AuthFailed.
    let server_final = read_string_frame(stream)?;
    if client.verify_server_final(&server_final).is_err() {
        // Drain the trailing status byte if the server sent one, then fail.
        let _ = read_status(stream);
        return Err(ProtocolError::AuthFailed(
            "server signature verification failed (wrong password or untrusted server)".into(),
        ));
    }

    read_status(stream)
}

// ---------------------------------------------------------------------------
// Message framing (request/response loop)
// ---------------------------------------------------------------------------

/// Read a framed message: `request_id(8) + payload_length(4) + JSON`.
pub fn read_message<S: Read>(stream: &mut S) -> Result<Message> {
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

/// Write a framed response message.
pub fn write_message<S: Write>(
    stream: &mut S,
    request_id: u64,
    payload: &serde_json::Value,
) -> Result<()> {
    let payload_bytes =
        serde_json::to_vec(payload).map_err(|e| ProtocolError::InvalidJson(e.to_string()))?;

    let mut buf = BytesMut::with_capacity(12 + payload_bytes.len());
    buf.put_u64(request_id);
    buf.put_u32(payload_bytes.len() as u32);
    buf.put_slice(&payload_bytes);

    stream.write_all(&buf)?;
    stream.flush()?;
    Ok(())
}
