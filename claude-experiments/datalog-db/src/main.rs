use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use tracing::info;

use datalog_db::backup::{spawn_backup_scheduler, BackupSchedulerConfig};
use datalog_db::db::Database;
use datalog_db::server::{BackupContext, Server};
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

    // Auth is required by default. The operator supplies a shared bearer
    // token via `--auth-token <secret>` or the `DATALOG_AUTH_TOKEN` env var
    // (the env var is preferred for secrets — it won't show up in `ps`).
    // Running without auth is allowed only with an explicit `--no-auth`
    // flag, so an unauthenticated server is always a deliberate choice and
    // never the result of a forgotten flag.
    let no_auth = args.iter().any(|a| a == "--no-auth");
    let auth_token: Option<Vec<u8>> = arg_value(&args, "--auth-token")
        .or_else(|| std::env::var("DATALOG_AUTH_TOKEN").ok())
        .filter(|t| !t.is_empty())
        .map(|t| t.into_bytes());

    if auth_token.is_none() && !no_auth {
        eprintln!(
            "error: no auth token configured.\n  \
             Set one with --auth-token <secret> or the DATALOG_AUTH_TOKEN env var,\n  \
             or pass --no-auth to run an unauthenticated server (open to anyone\n  \
             who can reach {}).",
            bind_addr
        );
        std::process::exit(1);
    }
    if auth_token.is_some() && no_auth {
        eprintln!("error: --no-auth conflicts with a configured auth token; pass only one.");
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

    info!("Starting server on {}", bind_addr);
    let server = Server::bind_with_auth(&bind_addr, db, backup_ctx, auth_token)?;
    server.run()?;

    Ok(())
}

fn default_data_dir() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .map(|home| home.join(".datalog-data"))
        .unwrap_or_else(|| PathBuf::from(".datalog-data"))
}
