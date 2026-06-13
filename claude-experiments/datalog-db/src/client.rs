//! Typed client library for datalog-db.
//!
//! Connects to a running `datalog-db` server over TCP, performs the
//! version handshake, and exposes one method per server operation:
//! [`define_type`], [`define_enum`], [`transact`], [`query`], [`explain`],
//! [`schema`], [`status`], and an escape-hatch [`send_raw`].
//!
//! See `docs/rust-client.md` for a longer-form guide (transaction-op JSON
//! shapes, query JSON shape, threading, error model).
//!
//! # Quick start
//!
//! Start a server in another process:
//!
//! ```sh
//! cargo run --release --bin datalog-db -- --data-dir ./data --bind 127.0.0.1:5557
//! ```
//!
//! Then in your code:
//!
//! ```no_run
//! use datalog_db::client::Client;
//! use datalog_db::schema::{EntityTypeDef, FieldDef, FieldType};
//! use serde_json::json;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let mut client = Client::connect("127.0.0.1:5557")?;
//!
//! // 1. Define an entity type (a "collection").
//! client.define_type(&EntityTypeDef {
//!     name: "User".into(),
//!     fields: vec![
//!         FieldDef { name: "name".into(),  field_type: FieldType::String,
//!                    required: true,  unique: false, indexed: false,
//!                    cardinality: Default::default(), fulltext: false, ann: false },
//!         FieldDef { name: "email".into(), field_type: FieldType::String,
//!                    required: true,  unique: true,  indexed: true,
//!                    cardinality: Default::default(), fulltext: false, ann: false  },
//!         FieldDef { name: "age".into(),   field_type: FieldType::I64,
//!                    required: false, unique: false, indexed: false,
//!                    cardinality: Default::default(), fulltext: false, ann: false },
//!     ],
//!     unique_keys: vec![],
//! })?;
//!
//! // 2. Insert an entity. `transact` takes raw JSON ops; one op per
//! //    assert/retract/retract_entity. The server returns the assigned
//! //    entity id.
//! let tx = client.transact(vec![
//!     json!({ "assert": "User", "data": { "name": "Alice", "email": "a@b.com", "age": 30 } }),
//! ])?;
//! let alice = tx.entity_ids[0];
//!
//! // 3. Update by including `entity` in the op.
//! client.transact(vec![
//!     json!({ "assert": "User", "entity": alice, "data": { "age": 31 } }),
//! ])?;
//!
//! // 4. Query. `find` lists the variables to return; `where` is one
//! //    clause per entity binding. Predicates: {"gt": N}, {"lt": N},
//! //    {"gte": N}, {"lte": N}, {"ne": V}.
//! let result = client.query(&json!({
//!     "find":  ["?u", "?name"],
//!     "where": [{ "bind": "?u", "type": "User", "name": "?name", "age": {"gt": 25} }],
//! }))?;
//! for row in &result.rows {
//!     println!("{}: {}", row[0], row[1]);
//! }
//!
//! // 5. Retract specific fields, or whole entities.
//! client.transact(vec![
//!     json!({ "retract": "User", "entity": alice, "fields": ["age"] }),
//! ])?;
//! # Ok(()) }
//! ```
//!
//! # Errors
//!
//! All methods return [`Result<T>`] = `Result<T, ClientError>`. Server-side
//! errors come back as [`ClientError::Server`] with the message from the
//! server (e.g. `"unique field 'email' must also be required"`). Wire- or
//! IO-level problems surface as [`ClientError::Protocol`].
//!
//! # Concurrency
//!
//! A `Client` owns one `TcpStream` and is **not** `Sync`. Use one client
//! per thread, or share an `Arc<Mutex<Client>>`. For higher throughput
//! spawn multiple connections — the server is multi-threaded and supports
//! `parallel_writes` plus incremental cache appends.
//!
//! [`define_type`]: Client::define_type
//! [`define_enum`]: Client::define_enum
//! [`transact`]:    Client::transact
//! [`query`]:       Client::query
//! [`explain`]:     Client::explain
//! [`schema`]:      Client::schema
//! [`status`]:      Client::status
//! [`send_raw`]:    Client::send_raw

use std::net::TcpStream;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::protocol;
use crate::schema::{EntityTypeDef, EnumTypeDef, FieldType};
use crate::transport::{self, ClientStream};

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

#[derive(Debug, thiserror::Error)]
pub enum ClientError {
    #[error("Protocol error: {0}")]
    Protocol(#[from] protocol::ProtocolError),

    #[error("Server error: {0}")]
    Server(String),

    #[error("Unexpected response: {0}")]
    UnexpectedResponse(String),
}

pub type Result<T> = std::result::Result<T, ClientError>;

// ---------------------------------------------------------------------------
// Response types
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub struct DefineResult {
    pub tx_id: u64,
}

#[derive(Debug)]
pub struct TransactResult {
    pub tx_id: u64,
    pub entity_ids: Vec<u64>,
    pub datom_count: usize,
    pub timestamp_ms: u64,
}

#[derive(Debug)]
pub struct QueryResult {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<serde_json::Value>>,
}

#[derive(Debug)]
pub struct ExplainResult {
    pub plan: serde_json::Value,
    pub display: String,
}

#[derive(Debug)]
pub struct SchemaResult {
    pub types: serde_json::Value,
    pub enums: serde_json::Value,
}

#[derive(Debug)]
pub struct StatusResult {
    pub server: String,
    pub version: String,
}

// ---------------------------------------------------------------------------
// Client
// ---------------------------------------------------------------------------

pub struct Client {
    stream: ClientStream,
    next_request_id: AtomicU64,
}

impl Client {
    /// Connect to a datalog-db server with no auth token (plaintext).
    ///
    /// Use this against a server started with auth disabled. If the server
    /// requires auth, the handshake fails with
    /// [`protocol::ProtocolError::AuthFailed`]. Use [`Client::connect_with_token`],
    /// [`Client::connect_tls`], or [`Client::connect_scram`] to authenticate.
    pub fn connect(addr: &str) -> Result<Self> {
        Self::connect_with_token(addr, "")
    }

    /// Connect (plaintext) presenting a shared bearer `token` at the handshake.
    pub fn connect_with_token(addr: &str, token: &str) -> Result<Self> {
        let mut stream = plain_stream(addr)?;
        protocol::client_handshake(&mut stream, token.as_bytes())?;
        Ok(Self::wrap(stream))
    }

    /// Connect over **TLS** presenting a shared bearer `token`.
    ///
    /// `ca_pem` is the CA (or self-signed server cert) to trust; `server_name`
    /// must match the server certificate's name. Defaults the name to the host
    /// part of `addr` via [`Client::connect_tls_with_token`].
    pub fn connect_tls(addr: &str, token: &str, ca_pem: &Path) -> Result<Self> {
        let server_name = host_of(addr);
        Self::connect_tls_with_token(addr, token, ca_pem, &server_name)
    }

    /// TLS + token with an explicit `server_name` for certificate validation.
    pub fn connect_tls_with_token(
        addr: &str,
        token: &str,
        ca_pem: &Path,
        server_name: &str,
    ) -> Result<Self> {
        let mut stream = tls_stream(addr, ca_pem, server_name)?;
        protocol::client_handshake(&mut stream, token.as_bytes())?;
        Ok(Self::wrap(stream))
    }

    /// Connect over **TLS** authenticating as `username`/`password` with
    /// per-user SCRAM-SHA-256. The server's signature is verified (mutual auth)
    /// before this returns `Ok`. `server_name` defaults to the host part of
    /// `addr`; use [`Client::connect_scram_with_name`] to override.
    pub fn connect_scram(
        addr: &str,
        username: &str,
        password: &str,
        ca_pem: &Path,
    ) -> Result<Self> {
        let server_name = host_of(addr);
        Self::connect_scram_with_name(addr, username, password, ca_pem, &server_name)
    }

    /// SCRAM over TLS with an explicit `server_name`.
    pub fn connect_scram_with_name(
        addr: &str,
        username: &str,
        password: &str,
        ca_pem: &Path,
        server_name: &str,
    ) -> Result<Self> {
        let mut stream = tls_stream(addr, ca_pem, server_name)?;
        protocol::client_handshake_scram(&mut stream, username, password)?;
        Ok(Self::wrap(stream))
    }

    /// SCRAM over **plaintext** (e.g. SCRAM on loopback without TLS). The
    /// password is protected by SCRAM's challenge-response — it never crosses
    /// the wire — but without TLS the rest of the session is unencrypted.
    pub fn connect_scram_plain(addr: &str, username: &str, password: &str) -> Result<Self> {
        let mut stream = plain_stream(addr)?;
        protocol::client_handshake_scram(&mut stream, username, password)?;
        Ok(Self::wrap(stream))
    }

    fn wrap(stream: ClientStream) -> Self {
        Self {
            stream,
            next_request_id: AtomicU64::new(1),
        }
    }

    /// Define an entity type from an `EntityTypeDef`.
    pub fn define_type(&mut self, type_def: &EntityTypeDef) -> Result<DefineResult> {
        let fields: Vec<serde_json::Value> = type_def
            .fields
            .iter()
            .map(|f| {
                serde_json::json!({
                    "name": f.name,
                    "type": field_type_to_wire(&f.field_type),
                    "required": f.required,
                    "unique": f.unique,
                    "indexed": f.indexed,
                })
            })
            .collect();

        let payload = serde_json::json!({
            "type": "define",
            "entity_type": type_def.name,
            "fields": fields,
        });

        let resp = self.send_and_check(payload)?;
        let data = resp
            .get("data")
            .ok_or_else(|| ClientError::UnexpectedResponse("missing 'data'".into()))?;
        let tx_id = data["tx_id"]
            .as_u64()
            .ok_or_else(|| ClientError::UnexpectedResponse("missing 'tx_id'".into()))?;

        Ok(DefineResult { tx_id })
    }

    /// Define an enum type from an `EnumTypeDef`.
    pub fn define_enum(&mut self, enum_def: &EnumTypeDef) -> Result<DefineResult> {
        let variants: Vec<serde_json::Value> = enum_def
            .variants
            .iter()
            .map(|v| {
                let fields: Vec<serde_json::Value> = v
                    .fields
                    .iter()
                    .map(|f| {
                        serde_json::json!({
                            "name": f.name,
                            "type": field_type_to_wire(&f.field_type),
                            "required": f.required,
                        })
                    })
                    .collect();
                serde_json::json!({
                    "name": v.name,
                    "fields": fields,
                })
            })
            .collect();

        let payload = serde_json::json!({
            "type": "define_enum",
            "enum_name": enum_def.name,
            "variants": variants,
        });

        let resp = self.send_and_check(payload)?;
        let data = resp
            .get("data")
            .ok_or_else(|| ClientError::UnexpectedResponse("missing 'data'".into()))?;
        let tx_id = data["tx_id"]
            .as_u64()
            .ok_or_else(|| ClientError::UnexpectedResponse("missing 'tx_id'".into()))?;

        Ok(DefineResult { tx_id })
    }

    /// Drop an entity type. `hard == false` is a soft drop (definition
    /// retracted, data and history preserved); `hard == true` purges the
    /// type and all its datoms. Returns the raw `data` object so callers
    /// can inspect `mode`, `entities_purged`, `dangling_refs`, etc.
    pub fn drop_type(&mut self, name: &str, hard: bool) -> Result<serde_json::Value> {
        let payload = serde_json::json!({
            "type": "drop_type",
            "name": name,
            "hard": hard,
        });
        let resp = self.send_and_check(payload)?;
        resp.get("data")
            .cloned()
            .ok_or_else(|| ClientError::UnexpectedResponse("missing 'data'".into()))
    }

    /// Drop an enum type. Soft/hard semantics mirror [`Client::drop_type`].
    pub fn drop_enum(&mut self, name: &str, hard: bool) -> Result<serde_json::Value> {
        let payload = serde_json::json!({
            "type": "drop_enum",
            "name": name,
            "hard": hard,
        });
        let resp = self.send_and_check(payload)?;
        resp.get("data")
            .cloned()
            .ok_or_else(|| ClientError::UnexpectedResponse("missing 'data'".into()))
    }

    /// Execute a transaction with a list of operation JSON values.
    ///
    /// Each element should be an assert / retract / retract_entity object,
    /// e.g. `json!({"assert": "User", "data": {"name": "Alice"}})`.
    pub fn transact(&mut self, ops: Vec<serde_json::Value>) -> Result<TransactResult> {
        let payload = serde_json::json!({
            "type": "transact",
            "ops": ops,
        });

        let resp = self.send_and_check(payload)?;
        let data = resp
            .get("data")
            .ok_or_else(|| ClientError::UnexpectedResponse("missing 'data'".into()))?;

        let tx_id = data["tx_id"]
            .as_u64()
            .ok_or_else(|| ClientError::UnexpectedResponse("missing 'tx_id'".into()))?;
        let entity_ids: Vec<u64> = data["entity_ids"]
            .as_array()
            .ok_or_else(|| ClientError::UnexpectedResponse("missing 'entity_ids'".into()))?
            .iter()
            .filter_map(|v| v.as_u64())
            .collect();
        let datom_count = data["datom_count"]
            .as_u64()
            .ok_or_else(|| ClientError::UnexpectedResponse("missing 'datom_count'".into()))?
            as usize;
        let timestamp_ms = data["timestamp_ms"]
            .as_u64()
            .ok_or_else(|| ClientError::UnexpectedResponse("missing 'timestamp_ms'".into()))?;

        Ok(TransactResult {
            tx_id,
            entity_ids,
            datom_count,
            timestamp_ms,
        })
    }

    /// Execute a query.  The JSON should have `find`, `where`, and optionally
    /// `as_of` / `as_of_time` fields (the `"type": "query"` wrapper is added
    /// automatically).
    pub fn query(&mut self, query_json: &serde_json::Value) -> Result<QueryResult> {
        let mut payload = query_json.clone();
        payload
            .as_object_mut()
            .ok_or_else(|| ClientError::UnexpectedResponse("query must be a JSON object".into()))?
            .insert("type".into(), serde_json::json!("query"));

        let resp = self.send_and_check(payload)?;
        let data = resp
            .get("data")
            .ok_or_else(|| ClientError::UnexpectedResponse("missing 'data'".into()))?;

        let columns: Vec<String> = data["columns"]
            .as_array()
            .ok_or_else(|| ClientError::UnexpectedResponse("missing 'columns'".into()))?
            .iter()
            .filter_map(|v| v.as_str().map(String::from))
            .collect();

        let rows: Vec<Vec<serde_json::Value>> = data["rows"]
            .as_array()
            .ok_or_else(|| ClientError::UnexpectedResponse("missing 'rows'".into()))?
            .iter()
            .filter_map(|row| {
                row.as_array().map(|r| r.clone())
            })
            .collect();

        Ok(QueryResult { columns, rows })
    }

    /// Explain a query: returns the query plan without executing.
    /// The JSON should have `find`, `where`, and optionally `as_of` / `as_of_time`
    /// fields (the `"type": "explain"` wrapper is added automatically).
    pub fn explain(&mut self, query_json: &serde_json::Value) -> Result<ExplainResult> {
        let mut payload = query_json.clone();
        payload
            .as_object_mut()
            .ok_or_else(|| ClientError::UnexpectedResponse("query must be a JSON object".into()))?
            .insert("type".into(), serde_json::json!("explain"));

        let resp = self.send_and_check(payload)?;
        let data = resp
            .get("data")
            .ok_or_else(|| ClientError::UnexpectedResponse("missing 'data'".into()))?;

        let plan = data
            .get("plan")
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        let display = data
            .get("display")
            .and_then(|d| d.as_str())
            .unwrap_or("")
            .to_string();

        Ok(ExplainResult { plan, display })
    }

    /// Fetch the current schema (types + enums).
    pub fn schema(&mut self) -> Result<SchemaResult> {
        let payload = serde_json::json!({"type": "schema"});
        let resp = self.send_and_check(payload)?;
        let data = resp
            .get("data")
            .ok_or_else(|| ClientError::UnexpectedResponse("missing 'data'".into()))?;

        Ok(SchemaResult {
            types: data.get("types").cloned().unwrap_or(serde_json::Value::Null),
            enums: data.get("enums").cloned().unwrap_or(serde_json::Value::Null),
        })
    }

    /// Get server status (name + version).
    pub fn status(&mut self) -> Result<StatusResult> {
        let payload = serde_json::json!({"type": "status"});
        let resp = self.send_and_check(payload)?;
        let data = resp
            .get("data")
            .ok_or_else(|| ClientError::UnexpectedResponse("missing 'data'".into()))?;

        let server = data["server"]
            .as_str()
            .unwrap_or("unknown")
            .to_string();
        let version = data["version"]
            .as_str()
            .unwrap_or("unknown")
            .to_string();

        Ok(StatusResult { server, version })
    }

    /// Send a raw JSON payload and return the full response.
    pub fn send_raw(&mut self, payload: serde_json::Value) -> Result<serde_json::Value> {
        let req_id = self.next_request_id.fetch_add(1, Ordering::Relaxed);
        protocol::write_message(&mut self.stream, req_id, &payload)?;
        let msg = protocol::read_message(&mut self.stream)?;
        Ok(msg.payload)
    }

    // -----------------------------------------------------------------------
    // Internal helpers
    // -----------------------------------------------------------------------

    /// Send a request and check the response status.  Returns the full
    /// response JSON on `"ok"`, or a `ClientError::Server` on `"error"`.
    fn send_and_check(&mut self, payload: serde_json::Value) -> Result<serde_json::Value> {
        let resp = self.send_raw(payload)?;

        match resp.get("status").and_then(|s| s.as_str()) {
            Some("ok") => Ok(resp),
            Some("error") => {
                let msg = resp
                    .get("error")
                    .and_then(|e| e.as_str())
                    .unwrap_or("unknown server error")
                    .to_string();
                Err(ClientError::Server(msg))
            }
            other => Err(ClientError::UnexpectedResponse(format!(
                "unexpected status: {:?}",
                other
            ))),
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Open a plain TCP stream with Nagle disabled. Nagle hurts request/response
/// loops (40ms delayed-ACK stalls), worst on loopback where RTT is microseconds.
fn plain_stream(addr: &str) -> Result<ClientStream> {
    let stream = TcpStream::connect(addr).map_err(protocol::ProtocolError::Io)?;
    let _ = stream.set_nodelay(true);
    Ok(ClientStream::Plain(stream))
}

/// Open a TLS stream: connect TCP, set nodelay on the raw socket (it carries
/// through the tunnel), then wrap in rustls validating against `ca_pem`.
fn tls_stream(addr: &str, ca_pem: &Path, server_name: &str) -> Result<ClientStream> {
    let sock = TcpStream::connect(addr).map_err(protocol::ProtocolError::Io)?;
    let _ = sock.set_nodelay(true);
    let config = transport::client_config(Some(ca_pem)).map_err(protocol::ProtocolError::Io)?;
    transport::connect_tls(sock, config, server_name).map_err(|e| protocol::ProtocolError::Io(e).into())
}

/// Extract the host part of an `addr` (`host:port`) for TLS cert validation.
/// IPv6 literals in brackets are returned without the brackets.
fn host_of(addr: &str) -> String {
    if let Some(rest) = addr.strip_prefix('[') {
        // [::1]:5557 → ::1
        if let Some(end) = rest.find(']') {
            return rest[..end].to_string();
        }
    }
    addr.rsplit_once(':')
        .map(|(host, _)| host.to_string())
        .unwrap_or_else(|| addr.to_string())
}

fn field_type_to_wire(ft: &FieldType) -> String {
    match ft {
        FieldType::String => "string".into(),
        FieldType::I64 => "i64".into(),
        FieldType::F64 => "f64".into(),
        FieldType::Bool => "bool".into(),
        FieldType::Bytes => "bytes".into(),
        FieldType::Ref(target) => format!("ref({})", target),
        FieldType::Enum(target) => format!("enum({})", target),
        FieldType::List(elem) => format!("[{}]", field_type_to_wire(elem)),
        FieldType::Vector(dim) => format!("vector({})", dim),
    }
}
