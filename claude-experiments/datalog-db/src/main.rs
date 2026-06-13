use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use tracing::info;

use datalog_db::auth::scram::Verifier;
use datalog_db::backup::{spawn_backup_scheduler, BackupSchedulerConfig};
use datalog_db::db::Database;
use datalog_db::server::{AuthConfig, BackupContext, Server};
use datalog_db::storage::rocksdb_backend::RocksDbStorage;

fn arg_value(args: &[String], key: &str) -> Option<String> {
    args.iter()
        .position(|a| a == key)
        .and_then(|i| args.get(i + 1))
        .cloned()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let args: Vec<String> = std::env::args().collect();

    let data_dir = arg_value(&args, "--data-dir")
        .map(PathBuf::from)
        .unwrap_or_else(default_data_dir);

    let bind_addr = arg_value(&args, "--bind").unwrap_or_else(|| "127.0.0.1:5557".to_string());

    // --- Authentication configuration ---
    //
    // At least one method must be enabled, unless `--no-auth` is given
    // explicitly — an unauthenticated server is always a deliberate choice.
    //   --auth-token <t> / DATALOG_AUTH_TOKEN : shared bearer token (loopback consumers)
    //   --scram                               : per-user SCRAM-SHA-256
    //   --no-auth                             : explicit opt-out
    // Token + SCRAM may both be on at once; clients pick per-connection.
    let no_auth = args.iter().any(|a| a == "--no-auth");
    let scram = args.iter().any(|a| a == "--scram");
    let auth_token: Option<Vec<u8>> = arg_value(&args, "--auth-token")
        .or_else(|| std::env::var("DATALOG_AUTH_TOKEN").ok())
        .filter(|t| !t.is_empty())
        .map(|t| t.into_bytes());

    let any_auth = auth_token.is_some() || scram;
    if !any_auth && !no_auth {
        eprintln!(
            "error: no authentication configured.\n  \
             Enable one (or both) of:\n    \
               --auth-token <secret>  (or DATALOG_AUTH_TOKEN env)  — shared token\n    \
               --scram                                            — per-user SCRAM\n  \
             or pass --no-auth to run an unauthenticated server (open to anyone\n  \
             who can reach {}).",
            bind_addr
        );
        std::process::exit(1);
    }
    if any_auth && no_auth {
        eprintln!("error: --no-auth conflicts with --auth-token/--scram; pass only one mode.");
        std::process::exit(1);
    }

    // --- TLS configuration ---
    // Both-or-neither, mirroring the auth validation. Plaintext is allowed
    // (loopback/dev); remote exposure should set both.
    let tls_cert = arg_value(&args, "--tls-cert").map(PathBuf::from);
    let tls_key = arg_value(&args, "--tls-key").map(PathBuf::from);
    if tls_cert.is_some() != tls_key.is_some() {
        eprintln!("error: --tls-cert and --tls-key must be given together.");
        std::process::exit(1);
    }

    // Optional periodic-backup configuration. `--backup-dir` is the
    // opt-in; the other two have sensible defaults.
    let backup_dir = arg_value(&args, "--backup-dir").map(PathBuf::from);
    let backup_interval_mins: u64 = arg_value(&args, "--backup-interval-mins")
        .map(|v| v.parse().expect("--backup-interval-mins must be a number"))
        .unwrap_or(60);
    let backup_retain: usize = arg_value(&args, "--backup-retain")
        .map(|v| v.parse().expect("--backup-retain must be a number"))
        .unwrap_or(24);

    info!("Opening database at {:?}", data_dir);
    std::fs::create_dir_all(&data_dir)?;

    let storage = RocksDbStorage::open(&data_dir)?;
    let storage = Arc::new(storage);

    let db = Database::open(storage)?;
    let db = Arc::new(db);

    // First-user bootstrap: when SCRAM is enabled, the store has no users, and
    // `--bootstrap-user` is given, create that user once from
    // DATALOG_BOOTSTRAP_PASSWORD (never argv). Lets a fresh containerized deploy
    // come up self-sufficient; for hands-on setups prefer `datalog user add`
    // offline. Idempotent: skipped once any user exists.
    if scram {
        if let Some(bootstrap_user) = arg_value(&args, "--bootstrap-user") {
            if db.count_users()? == 0 {
                let password = std::env::var("DATALOG_BOOTSTRAP_PASSWORD").unwrap_or_default();
                if password.is_empty() {
                    eprintln!(
                        "error: --bootstrap-user requires DATALOG_BOOTSTRAP_PASSWORD (non-empty)."
                    );
                    std::process::exit(1);
                }
                db.put_scram_verifier(&bootstrap_user, &Verifier::create(&password))?;
                info!("Bootstrap SCRAM user created: {}", bootstrap_user);
            } else {
                info!("--bootstrap-user ignored: user store is not empty");
            }
        }
    }

    // Spawn the backup scheduler before the server so the first scheduled
    // tick is anchored to startup. The handle is held in `_scheduler` for
    // the whole main scope so the background thread stays alive.
    let (backup_ctx, _scheduler) = match backup_dir {
        Some(root) => {
            info!(
                "Auto-backups enabled: root={:?} interval={}min retain={}",
                root, backup_interval_mins, backup_retain
            );
            let handle = spawn_backup_scheduler(
                db.clone(),
                BackupSchedulerConfig {
                    root: root.clone(),
                    interval: Duration::from_secs(backup_interval_mins * 60),
                    retain: backup_retain,
                },
            );
            (
                Some(BackupContext {
                    root,
                    retain: backup_retain,
                }),
                Some(handle),
            )
        }
        None => (None, None),
    };

    // Build TLS config if cert/key were provided.
    let tls = match (&tls_cert, &tls_key) {
        (Some(cert), Some(key)) => {
            info!("TLS enabled: cert={:?} key={:?}", cert, key);
            Some(datalog_db::transport::server_config(cert, key)?)
        }
        _ => None,
    };

    let auth = AuthConfig {
        token: auth_token,
        scram,
    };

    info!("Starting server on {}", bind_addr);
    let server = Server::bind_full(&bind_addr, db, backup_ctx, auth, tls)?;
    server.run()?;

    Ok(())
}

fn default_data_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .map(|home| home.join(".datalog-data"))
        .unwrap_or_else(|| PathBuf::from(".datalog-data"))
}
