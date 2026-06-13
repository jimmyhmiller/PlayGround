//! A tiny, pure-Rust client for the `datalog-db` wire protocol.
//!
//! The protocol (verified against the live server):
//! - **Handshake (VERSION 2):** client sends `MAGIC` (`u32` BE), `VERSION`
//!   (`u32` BE), then an auth frame `len(u32 BE) + token bytes`; server replies
//!   with a single status byte (`0x00` = OK, else an error frame
//!   `len(u32 BE) + utf8` follows). The token comes from `DATALOG_AUTH_TOKEN`.
//! - **Frames (both directions):** `request_id` (`u64` BE) + `len` (`u32` BE) +
//!   `len` bytes of JSON. The server echoes the same `request_id` back.
//!
//! Requests are JSON objects with a top-level `"type"` (`"query"`, `"transact"`,
//! `"schema"`, …) and the rest of the request's fields as siblings (NOT nested
//! under a `payload` key — that was a real footgun). A query carries `find` /
//! `where` / `order_by` / `having` / `limit` at the top level.
//!
//! This crate is deliberately small and synchronous: a fresh TCP connection per
//! [`Client`], blocking reads, `serde_json` for bodies. It is shared by the
//! Axiom reloader and the gatekeeper analytics function.

use std::io::{self, Read, Write};
use std::net::TcpStream;
use std::time::Duration;

use serde_json::{json, Value};

const MAGIC: u32 = 0xDA7A_1061;
// VERSION 2 added a shared-token auth frame to the handshake (4-byte length +
// token bytes) sent right after magic/version. Always present, even when the
// token is empty (which only authenticates against a --no-auth server).
const VERSION: u32 = 2;

/// A connected datalog-db client. One owns a single TCP connection and a
/// monotonic request-id counter.
pub struct Client {
    stream: TcpStream,
    next_id: u64,
}

/// The result of a `query`: column names plus row data (each cell a JSON value).
#[derive(Debug, Clone)]
pub struct QueryResult {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<Value>>,
}

impl QueryResult {
    /// The single scalar in a one-cell result (e.g. a `count`), or `None` if the
    /// shape isn't exactly one row × one column.
    pub fn scalar(&self) -> Option<&Value> {
        match self.rows.as_slice() {
            [row] => match row.as_slice() {
                [v] => Some(v),
                _ => None,
            },
            _ => None,
        }
    }
}

/// Errors talking to datalog-db: an I/O failure, a malformed frame, or an error
/// status returned by the server for a request.
#[derive(Debug)]
pub enum Error {
    Io(io::Error),
    Protocol(String),
    /// The server processed the request but returned `{"status":"error", ...}`.
    Server(String),
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Io(e) => write!(f, "datalog io error: {e}"),
            Error::Protocol(m) => write!(f, "datalog protocol error: {m}"),
            Error::Server(m) => write!(f, "datalog server error: {m}"),
        }
    }
}
impl std::error::Error for Error {}
impl From<io::Error> for Error {
    fn from(e: io::Error) -> Self {
        Error::Io(e)
    }
}

type Result<T> = std::result::Result<T, Error>;

impl Client {
    /// Connect to `addr` (e.g. `"127.0.0.1:5557"`) and complete the handshake.
    ///
    /// The auth token is read from `DATALOG_AUTH_TOKEN` (empty if unset, which
    /// only works against a `--no-auth` server). Use [`Client::connect_with_token`]
    /// to pass it explicitly.
    pub fn connect(addr: &str) -> Result<Client> {
        let token = std::env::var("DATALOG_AUTH_TOKEN").unwrap_or_default();
        Self::connect_with_token(addr, &token)
    }

    /// Connect to `addr`, presenting `token` at the handshake.
    pub fn connect_with_token(addr: &str, token: &str) -> Result<Client> {
        let stream = TcpStream::connect(addr)?;
        stream.set_read_timeout(Some(Duration::from_secs(120)))?;
        stream.set_write_timeout(Some(Duration::from_secs(120)))?;
        let mut c = Client { stream, next_id: 0 };
        c.handshake(token.as_bytes())?;
        Ok(c)
    }

    fn handshake(&mut self, token: &[u8]) -> Result<()> {
        self.stream.write_all(&MAGIC.to_be_bytes())?;
        self.stream.write_all(&VERSION.to_be_bytes())?;
        // Auth frame: 4-byte big-endian length + token bytes.
        self.stream.write_all(&(token.len() as u32).to_be_bytes())?;
        self.stream.write_all(token)?;
        self.stream.flush()?;
        let mut status = [0u8; 1];
        self.read_exact(&mut status)?;
        if status[0] == 0x00 {
            return Ok(());
        }
        // Non-OK: an error frame (len + utf8 message) follows.
        let mut len_buf = [0u8; 4];
        self.read_exact(&mut len_buf)?;
        let len = u32::from_be_bytes(len_buf) as usize;
        let mut msg = vec![0u8; len];
        self.read_exact(&mut msg)?;
        Err(Error::Protocol(format!(
            "handshake rejected: {}",
            String::from_utf8_lossy(&msg)
        )))
    }

    /// Send one request object and read the response object. Verifies the echoed
    /// request id and surfaces `{"status":"error"}` as [`Error::Server`].
    fn request(&mut self, req: &Value) -> Result<Value> {
        self.next_id += 1;
        let id = self.next_id;
        let body = serde_json::to_vec(req).map_err(|e| Error::Protocol(e.to_string()))?;

        let mut frame = Vec::with_capacity(12 + body.len());
        frame.extend_from_slice(&id.to_be_bytes());
        frame.extend_from_slice(&(body.len() as u32).to_be_bytes());
        frame.extend_from_slice(&body);
        self.stream.write_all(&frame)?;
        self.stream.flush()?;

        let mut hdr = [0u8; 12];
        self.read_exact(&mut hdr)?;
        let resp_id = u64::from_be_bytes(hdr[0..8].try_into().unwrap());
        let len = u32::from_be_bytes(hdr[8..12].try_into().unwrap()) as usize;
        if resp_id != id {
            return Err(Error::Protocol(format!(
                "response id {resp_id} != request id {id}"
            )));
        }
        let mut buf = vec![0u8; len];
        self.read_exact(&mut buf)?;
        let resp: Value =
            serde_json::from_slice(&buf).map_err(|e| Error::Protocol(e.to_string()))?;

        if resp.get("status").and_then(|s| s.as_str()) == Some("error") {
            let msg = resp
                .get("error")
                .and_then(|e| e.as_str())
                .unwrap_or("unknown error")
                .to_string();
            return Err(Error::Server(msg));
        }
        Ok(resp)
    }

    /// Run a query. `query` is the full query object MINUS the `"type"` field
    /// (e.g. `{"find":[...], "where":[...], "limit":10}`); this method adds the
    /// type. Returns the columns + rows.
    pub fn query(&mut self, mut query: Value) -> Result<QueryResult> {
        if let Some(obj) = query.as_object_mut() {
            obj.insert("type".into(), json!("query"));
        } else {
            return Err(Error::Protocol("query must be a JSON object".into()));
        }
        let resp = self.request(&query)?;
        let data = resp
            .get("data")
            .ok_or_else(|| Error::Protocol("query response missing 'data'".into()))?;
        let columns = data
            .get("columns")
            .and_then(|c| c.as_array())
            .map(|a| {
                a.iter()
                    .map(|v| v.as_str().unwrap_or("").to_string())
                    .collect()
            })
            .unwrap_or_default();
        let rows = data
            .get("rows")
            .and_then(|r| r.as_array())
            .map(|a| {
                a.iter()
                    .map(|row| row.as_array().cloned().unwrap_or_default())
                    .collect()
            })
            .unwrap_or_default();
        Ok(QueryResult { columns, rows })
    }

    /// Transact a batch of operations (e.g. `assert`s). `ops` is the array that
    /// goes under `"ops"`. Returns the raw response object on success.
    pub fn transact(&mut self, ops: Vec<Value>) -> Result<Value> {
        self.request(&json!({"type": "transact", "ops": ops}))
    }

    /// Convenience: the maximum value of an `i64` field across all entities of a
    /// type, or `None` if there are no entities. Used as a high-water-mark.
    pub fn max_i64(&mut self, type_name: &str, field: &str) -> Result<Option<i64>> {
        let res = self.query(json!({
            "find": [{"agg": "max", "var": "?v"}],
            "where": [{"bind": "?e", "type": type_name, field: "?v"}],
        }))?;
        match res.scalar() {
            Some(Value::Number(n)) => Ok(n.as_i64()),
            // No rows / null -> empty table.
            _ => Ok(None),
        }
    }

    fn read_exact(&mut self, buf: &mut [u8]) -> Result<()> {
        self.stream.read_exact(buf).map_err(Error::Io)
    }
}
