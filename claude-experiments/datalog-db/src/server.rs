use std::sync::Arc;

use tokio::net::{TcpListener, TcpStream};
use tracing::{info, warn};

use crate::db::{self, Database};
use crate::protocol::{self, Message};
use crate::query::Query;
use crate::tx::TxOp;

pub struct Server {
    db: Arc<Database>,
    listener: TcpListener,
}

impl Server {
    pub async fn bind(addr: &str, db: Arc<Database>) -> std::io::Result<Self> {
        let listener = TcpListener::bind(addr).await?;
        info!("Server listening on {}", addr);
        Ok(Self { db, listener })
    }

    pub fn local_addr(&self) -> std::io::Result<std::net::SocketAddr> {
        self.listener.local_addr()
    }

    pub async fn run(self) -> std::io::Result<()> {
        loop {
            let (stream, addr) = self.listener.accept().await?;
            info!("New connection from {}", addr);
            let db = self.db.clone();
            tokio::spawn(async move {
                if let Err(e) = handle_connection(stream, db).await {
                    warn!("Connection error from {}: {}", addr, e);
                }
                info!("Connection closed: {}", addr);
            });
        }
    }
}

async fn handle_connection(
    mut stream: TcpStream,
    db: Arc<Database>,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Handshake
    protocol::server_handshake(&mut stream).await?;

    // Message loop
    loop {
        let msg = match protocol::read_message(&mut stream).await {
            Ok(msg) => msg,
            Err(protocol::ProtocolError::Io(e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                // Client disconnected
                return Ok(());
            }
            Err(e) => return Err(e.into()),
        };

        let response = dispatch(&db, &msg).await;
        protocol::write_message(&mut stream, msg.request_id, &response).await?;
    }
}

async fn dispatch(db: &Database, msg: &Message) -> serde_json::Value {
    let request_type = msg
        .payload
        .get("type")
        .and_then(|t| t.as_str())
        .unwrap_or("");

    match request_type {
        "define" => handle_define(db, &msg.payload).await,
        "define_enum" => handle_define_enum(db, &msg.payload).await,
        "transact" => handle_transact(db, &msg.payload).await,
        "query" => handle_query(db, &msg.payload).await,
        "status" => handle_status(),
        other => serde_json::json!({
            "status": "error",
            "error": format!("unknown request type: '{}'", other)
        }),
    }
}

async fn handle_define(db: &Database, payload: &serde_json::Value) -> serde_json::Value {
    match db::parse_define_request(payload) {
        Ok(type_def) => {
            let name = type_def.name.clone();
            match db.define_type(type_def).await {
                Ok(tx_id) => serde_json::json!({
                    "status": "ok",
                    "data": {
                        "tx_id": tx_id,
                        "entity_type": name
                    }
                }),
                Err(e) => serde_json::json!({
                    "status": "error",
                    "error": e.to_string()
                }),
            }
        }
        Err(e) => serde_json::json!({
            "status": "error",
            "error": e
        }),
    }
}

async fn handle_define_enum(db: &Database, payload: &serde_json::Value) -> serde_json::Value {
    match db::parse_define_enum_request(payload) {
        Ok(enum_def) => {
            let name = enum_def.name.clone();
            match db.define_enum(enum_def).await {
                Ok(tx_id) => serde_json::json!({
                    "status": "ok",
                    "data": {
                        "tx_id": tx_id,
                        "enum_name": name
                    }
                }),
                Err(e) => serde_json::json!({
                    "status": "error",
                    "error": e.to_string()
                }),
            }
        }
        Err(e) => serde_json::json!({
            "status": "error",
            "error": e
        }),
    }
}

async fn handle_transact(db: &Database, payload: &serde_json::Value) -> serde_json::Value {
    let ops_json = match payload.get("ops").and_then(|o| o.as_array()) {
        Some(ops) => ops,
        None => {
            return serde_json::json!({
                "status": "error",
                "error": "missing 'ops' array"
            })
        }
    };

    let mut ops = Vec::new();
    for op_json in ops_json {
        match TxOp::from_json(op_json) {
            Ok(op) => ops.push(op),
            Err(e) => {
                return serde_json::json!({
                    "status": "error",
                    "error": e
                })
            }
        }
    }

    match db.transact(ops).await {
        Ok(result) => serde_json::json!({
            "status": "ok",
            "data": {
                "tx_id": result.tx_id,
                "entity_ids": result.entity_ids,
                "datom_count": result.datom_count
            }
        }),
        Err(e) => serde_json::json!({
            "status": "error",
            "error": e.to_string()
        }),
    }
}

async fn handle_query(db: &Database, payload: &serde_json::Value) -> serde_json::Value {
    let query = match Query::from_json(payload) {
        Ok(q) => q,
        Err(e) => {
            return serde_json::json!({
                "status": "error",
                "error": e
            })
        }
    };

    match db.query(&query).await {
        Ok(result) => {
            let rows: Vec<Vec<serde_json::Value>> = result
                .rows
                .iter()
                .map(|row| row.iter().map(value_to_json).collect())
                .collect();

            serde_json::json!({
                "status": "ok",
                "data": {
                    "columns": result.columns,
                    "rows": rows
                }
            })
        }
        Err(e) => serde_json::json!({
            "status": "error",
            "error": e.to_string()
        }),
    }
}

fn handle_status() -> serde_json::Value {
    serde_json::json!({
        "status": "ok",
        "data": {
            "server": "datalog-db",
            "version": env!("CARGO_PKG_VERSION")
        }
    })
}

fn value_to_json(v: &crate::datom::Value) -> serde_json::Value {
    match v {
        crate::datom::Value::String(s) => serde_json::Value::String(s.clone()),
        crate::datom::Value::I64(n) => serde_json::json!(n),
        crate::datom::Value::F64(n) => serde_json::json!(n),
        crate::datom::Value::Bool(b) => serde_json::json!(b),
        crate::datom::Value::Ref(id) => serde_json::json!({"ref": id}),
        crate::datom::Value::Bytes(b) => {
            serde_json::json!({"bytes": base64_encode(b)})
        }
        crate::datom::Value::Enum { variant, fields } => {
            let field_json: serde_json::Map<_, _> = fields
                .iter()
                .map(|(k, v)| (k.clone(), value_to_json(v)))
                .collect();
            serde_json::json!({ variant: field_json })
        }
    }
}

fn base64_encode(data: &[u8]) -> String {
    // Simple base64 encoding without external dependency
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut result = String::new();
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = if chunk.len() > 1 { chunk[1] as u32 } else { 0 };
        let b2 = if chunk.len() > 2 { chunk[2] as u32 } else { 0 };
        let triple = (b0 << 16) | (b1 << 8) | b2;
        result.push(CHARS[((triple >> 18) & 0x3F) as usize] as char);
        result.push(CHARS[((triple >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            result.push(CHARS[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
        if chunk.len() > 2 {
            result.push(CHARS[(triple & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
    }
    result
}
