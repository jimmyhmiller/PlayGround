//! Typed client library for datalog-db.
//!
//! Connects over TCP using the datalog-db wire protocol and provides
//! ergonomic methods for every server operation.

use std::net::TcpStream;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::protocol;
use crate::schema::{EntityTypeDef, EnumTypeDef, FieldType};

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
    stream: TcpStream,
    next_request_id: AtomicU64,
}

impl Client {
    /// Connect to a datalog-db server, performing the handshake.
    pub fn connect(addr: &str) -> Result<Self> {
        let mut stream = TcpStream::connect(addr)
            .map_err(|e| protocol::ProtocolError::Io(e))?;
        protocol::client_handshake(&mut stream)?;
        Ok(Self {
            stream,
            next_request_id: AtomicU64::new(1),
        })
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

fn field_type_to_wire(ft: &FieldType) -> String {
    match ft {
        FieldType::String => "string".into(),
        FieldType::I64 => "i64".into(),
        FieldType::F64 => "f64".into(),
        FieldType::Bool => "bool".into(),
        FieldType::Bytes => "bytes".into(),
        FieldType::Ref(target) => format!("ref({})", target),
        FieldType::Enum(target) => format!("enum({})", target),
    }
}
