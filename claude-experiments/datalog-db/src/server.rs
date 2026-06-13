use std::net::TcpListener;
use std::path::PathBuf;
use std::sync::Arc;

use tracing::{info, warn};

use crate::backup;
use crate::db::{self, Database};
use crate::protocol::{self, AuthOutcome, Message, ServerAuth, UserStore};
use crate::query::Query;
use crate::transport::ServerStream;
use crate::tx::TxOp;

/// Adapter letting the protocol layer look up SCRAM verifiers without depending
/// on `db`. Holds the same `Arc<Database>` the server already owns.
struct DbUserStore(Arc<Database>);

impl UserStore for DbUserStore {
    fn lookup(&self, username: &str) -> std::result::Result<Option<crate::auth::scram::Verifier>, String> {
        self.0.get_scram_verifier(username).map_err(|e| e.to_string())
    }
}

/// What the server accepts at the handshake. At least one method must be
/// enabled (enforced by `main`); both can be on at once (token for loopback
/// consumers, SCRAM for remote users), selected per-connection by the client.
#[derive(Clone, Default)]
pub struct AuthConfig {
    /// Shared bearer token. `Some` enables the `token` method.
    pub token: Option<Vec<u8>>,
    /// Enable per-user SCRAM (verifiers come from the `Database`).
    pub scram: bool,
}

/// Server-only random key for deriving deterministic mock verifiers for unknown
/// SCRAM users (enumeration resistance). Generated once at startup; never
/// persisted — it only needs to be stable for the lifetime of the process.
fn make_mock_key() -> Arc<Vec<u8>> {
    let mut k = vec![0u8; 32];
    crate::auth::scram::fill_random(&mut k);
    Arc::new(k)
}

/// Per-server backup state shared with request handlers. When `None`,
/// `backup_now` / `backup_list` return an error explaining that backups
/// were not configured at server startup. `retain` applies to both the
/// scheduler and on-demand `backup_now` calls so retention is a property
/// of the root, not of who took the backup.
#[derive(Clone)]
pub struct BackupContext {
    pub root: PathBuf,
    pub retain: usize,
}

pub struct Server {
    db: Arc<Database>,
    backup: Option<BackupContext>,
    listener: TcpListener,
    auth: AuthConfig,
    mock_key: Arc<Vec<u8>>,
    /// rustls config when TLS is enabled. `None` = plaintext (loopback/dev).
    tls: Option<Arc<rustls::ServerConfig>>,
}

impl Server {
    pub fn bind(addr: &str, db: Arc<Database>) -> std::io::Result<Self> {
        Self::bind_with(addr, db, None)
    }

    pub fn bind_with(
        addr: &str,
        db: Arc<Database>,
        backup: Option<BackupContext>,
    ) -> std::io::Result<Self> {
        Self::bind_full(addr, db, backup, AuthConfig::default(), None)
    }

    /// Back-compat shim: bind with only a shared token (or none).
    pub fn bind_with_auth(
        addr: &str,
        db: Arc<Database>,
        backup: Option<BackupContext>,
        auth_token: Option<Vec<u8>>,
    ) -> std::io::Result<Self> {
        Self::bind_full(
            addr,
            db,
            backup,
            AuthConfig {
                token: auth_token,
                scram: false,
            },
            None,
        )
    }

    /// Bind with full auth + optional TLS config.
    pub fn bind_full(
        addr: &str,
        db: Arc<Database>,
        backup: Option<BackupContext>,
        auth: AuthConfig,
        tls: Option<Arc<rustls::ServerConfig>>,
    ) -> std::io::Result<Self> {
        let listener = TcpListener::bind(addr)?;
        let transport = if tls.is_some() { "TLS" } else { "plaintext" };
        let mut methods = Vec::new();
        if auth.token.is_some() {
            methods.push("token");
        }
        if auth.scram {
            methods.push("scram");
        }
        let methods = if methods.is_empty() {
            "none (open)".to_string()
        } else {
            methods.join("+")
        };
        info!(
            "Server listening on {} ({transport}, auth: {methods})",
            addr
        );
        Ok(Self {
            db,
            backup,
            listener,
            auth,
            mock_key: make_mock_key(),
            tls,
        })
    }

    pub fn local_addr(&self) -> std::io::Result<std::net::SocketAddr> {
        self.listener.local_addr()
    }

    pub fn run(self) -> std::io::Result<()> {
        for stream in self.listener.incoming() {
            let stream = stream?;
            let addr = stream.peer_addr()?;
            // Set socket options on the RAW socket before any TLS wrapping —
            // they call straight through and keep applying inside the tunnel.
            let _ = stream.set_nodelay(true);
            info!("New connection from {}", addr);

            // Wrap in TLS if configured. A failure to start the TLS session is
            // logged and the connection dropped (don't fall back to plaintext).
            let conn = match &self.tls {
                None => ServerStream::Plain(stream),
                Some(cfg) => match ServerStream::accept_tls(stream, cfg.clone()) {
                    Ok(c) => c,
                    Err(e) => {
                        warn!("TLS setup failed for {}: {}", addr, e);
                        continue;
                    }
                },
            };

            let db = self.db.clone();
            let backup = self.backup.clone();
            let auth = self.auth.clone();
            let mock_key = self.mock_key.clone();
            std::thread::spawn(move || {
                if let Err(e) = handle_connection(conn, db, backup, auth, mock_key) {
                    warn!("Connection error from {}: {}", addr, e);
                }
                info!("Connection closed: {}", addr);
            });
        }
        Ok(())
    }
}

fn handle_connection(
    mut stream: ServerStream,
    db: Arc<Database>,
    backup: Option<BackupContext>,
    auth: AuthConfig,
    mock_key: Arc<Vec<u8>>,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    // Build the per-connection auth view. The user store is only consulted when
    // SCRAM is enabled.
    let user_store = DbUserStore(db.clone());
    // Open mode (accept any token) only when no method is configured — i.e. the
    // operator passed --no-auth. With a token or SCRAM enabled, this is false.
    let allow_no_auth = auth.token.is_none() && !auth.scram;
    let server_auth = ServerAuth {
        expected_token: auth.token.clone(),
        users: if auth.scram {
            Some(&user_store as &dyn UserStore)
        } else {
            None
        },
        mock_key: &mock_key,
        allow_no_auth,
    };

    // Handshake: authenticates the client (token or SCRAM) and yields identity.
    let AuthOutcome { user } = protocol::server_handshake(&mut stream, &server_auth)?;
    if let Some(ref u) = user {
        info!("Authenticated SCRAM user: {}", u);
    }

    // Message loop
    loop {
        let msg = match protocol::read_message(&mut stream) {
            Ok(msg) => msg,
            Err(protocol::ProtocolError::Io(e))
                if e.kind() == std::io::ErrorKind::UnexpectedEof =>
            {
                // Client disconnected (incl. a clean TLS close_notify → EOF).
                return Ok(());
            }
            Err(e) => return Err(e.into()),
        };

        let response = dispatch(&db, backup.as_ref(), &msg);
        protocol::write_message(&mut stream, msg.request_id, &response)?;
    }
}

fn dispatch(db: &Database, backup: Option<&BackupContext>, msg: &Message) -> serde_json::Value {
    let request_type = msg
        .payload
        .get("type")
        .and_then(|t| t.as_str())
        .unwrap_or("");

    match request_type {
        "define" => handle_define(db, &msg.payload),
        "define_enum" => handle_define_enum(db, &msg.payload),
        "drop_type" => handle_drop(db, &msg.payload, false),
        "drop_enum" => handle_drop(db, &msg.payload, true),
        "transact" => handle_transact(db, &msg.payload),
        "query" => handle_query(db, &msg.payload),
        "explain" => handle_explain(db, &msg.payload),
        "schema" => handle_schema(db),
        "status" => handle_status(),
        "backup_now" => handle_backup_now(db, backup),
        "backup_list" => handle_backup_list(backup),
        other => serde_json::json!({
            "status": "error",
            "error": format!("unknown request type: '{}'", other)
        }),
    }
}

fn handle_backup_now(db: &Database, backup: Option<&BackupContext>) -> serde_json::Value {
    let ctx = match backup {
        Some(b) => b,
        None => {
            return serde_json::json!({
                "status": "error",
                "error": "backups not configured; start server with --backup-dir"
            })
        }
    };

    match backup::create_checkpoint(db, &ctx.root) {
        Ok(path) => {
            let pruned = backup::prune_checkpoints(&ctx.root, ctx.retain).unwrap_or(0);
            serde_json::json!({
                "status": "ok",
                "data": {
                    "path": path.display().to_string(),
                    "pruned": pruned,
                }
            })
        }
        Err(e) => serde_json::json!({
            "status": "error",
            "error": format!("backup failed: {}", e)
        }),
    }
}

fn handle_backup_list(backup: Option<&BackupContext>) -> serde_json::Value {
    let ctx = match backup {
        Some(b) => b,
        None => {
            return serde_json::json!({
                "status": "error",
                "error": "backups not configured; start server with --backup-dir"
            })
        }
    };

    match backup::list_checkpoints(&ctx.root) {
        Ok(paths) => {
            let entries: Vec<serde_json::Value> = paths
                .iter()
                .map(|p| {
                    let name = p
                        .file_name()
                        .and_then(|n| n.to_str())
                        .unwrap_or("")
                        .to_string();
                    let timestamp_ms: u64 = name
                        .get(..13)
                        .and_then(|s| s.parse().ok())
                        .unwrap_or(0);
                    serde_json::json!({
                        "path": p.display().to_string(),
                        "name": name,
                        "timestamp_ms": timestamp_ms,
                    })
                })
                .collect();
            serde_json::json!({
                "status": "ok",
                "data": {
                    "root": ctx.root.display().to_string(),
                    "backups": entries,
                }
            })
        }
        Err(e) => serde_json::json!({
            "status": "error",
            "error": format!("list failed: {}", e)
        }),
    }
}

fn handle_define(db: &Database, payload: &serde_json::Value) -> serde_json::Value {
    match db::parse_define_request(payload) {
        Ok(type_def) => {
            let name = type_def.name.clone();
            match db.define_type(type_def) {
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

fn handle_define_enum(db: &Database, payload: &serde_json::Value) -> serde_json::Value {
    match db::parse_define_enum_request(payload) {
        Ok(enum_def) => {
            let name = enum_def.name.clone();
            match db.define_enum(enum_def) {
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

/// Drop (soft) or purge (hard) a type or enum. `is_enum` selects which.
/// The request carries `{"name": "<Name>", "hard": <bool>}`; `hard`
/// defaults to false (soft drop).
fn handle_drop(db: &Database, payload: &serde_json::Value, is_enum: bool) -> serde_json::Value {
    let name = match payload.get("name").and_then(|n| n.as_str()) {
        Some(n) => n,
        None => {
            return serde_json::json!({
                "status": "error",
                "error": "missing 'name'"
            })
        }
    };
    let hard = payload.get("hard").and_then(|h| h.as_bool()).unwrap_or(false);

    let outcome = if is_enum {
        db.drop_enum(name, hard)
    } else {
        db.drop_type(name, hard)
    };

    match outcome {
        Ok(r) => serde_json::json!({
            "status": "ok",
            "data": {
                "dropped": r.name,
                "kind": r.kind,
                "mode": if r.hard { "hard" } else { "soft" },
                "tx_id": r.tx_id,
                "entities_purged": r.entities_purged,
                "datoms_deleted": r.datoms_deleted,
                "dangling_refs": r.dangling_refs,
                "warnings": r.warnings,
            }
        }),
        Err(e) => serde_json::json!({
            "status": "error",
            "error": e.to_string()
        }),
    }
}

fn handle_transact(db: &Database, payload: &serde_json::Value) -> serde_json::Value {
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

    match db.transact(ops) {
        Ok(result) => serde_json::json!({
            "status": "ok",
            "data": {
                "tx_id": result.tx_id,
                "entity_ids": result.entity_ids,
                "datom_count": result.datom_count,
                "timestamp_ms": result.timestamp_ms
            }
        }),
        Err(e) => serde_json::json!({
            "status": "error",
            "error": e.to_string()
        }),
    }
}

fn handle_query(db: &Database, payload: &serde_json::Value) -> serde_json::Value {
    let query = match Query::from_json(payload) {
        Ok(q) => q,
        Err(e) => {
            return serde_json::json!({
                "status": "error",
                "error": e
            })
        }
    };

    // If explain flag is set, return the plan instead of executing
    if query.explain {
        return handle_explain(db, payload);
    }

    match db.query(&query) {
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

fn handle_explain(db: &Database, payload: &serde_json::Value) -> serde_json::Value {
    let query = match Query::from_json(payload) {
        Ok(q) => q,
        Err(e) => {
            return serde_json::json!({
                "status": "error",
                "error": e
            })
        }
    };

    match db.explain(&query) {
        Ok(plan) => {
            serde_json::json!({
                "status": "ok",
                "data": {
                    "plan": plan.to_json(),
                    "display": format!("{}", plan)
                }
            })
        }
        Err(e) => serde_json::json!({
            "status": "error",
            "error": e.to_string()
        }),
    }
}

fn handle_schema(db: &Database) -> serde_json::Value {
    serde_json::json!({
        "status": "ok",
        "data": db.schema_json()
    })
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
        crate::datom::Value::String(s) => serde_json::Value::String(s.to_string()),
        crate::datom::Value::I64(n) => serde_json::json!(n),
        crate::datom::Value::F64(n) => serde_json::json!(n),
        crate::datom::Value::Bool(b) => serde_json::json!(b),
        crate::datom::Value::Ref(id) => serde_json::json!({"ref": id}),
        crate::datom::Value::Bytes(b) => {
            serde_json::json!({"bytes": base64_encode(b)})
        }
        crate::datom::Value::Enum(e) => { let variant = &e.variant; let fields = &e.fields;
            let field_json: serde_json::Map<_, _> = fields
                .iter()
                .map(|(k, v)| (k.clone(), value_to_json(v)))
                .collect();
            serde_json::json!({ variant: field_json })
        }
        crate::datom::Value::List(items) => {
            serde_json::Value::Array(items.iter().map(value_to_json).collect())
        }
        crate::datom::Value::Vector(v) => {
            // Emit as {"vec": [...]} — the same tagged form the parser accepts,
            // so vectors round-trip through the wire.
            serde_json::json!({ "vec": v.iter().map(|f| *f as f64).collect::<Vec<_>>() })
        }
        crate::datom::Value::Null => serde_json::Value::Null,
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
