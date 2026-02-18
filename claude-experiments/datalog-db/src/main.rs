use std::path::PathBuf;
use std::sync::Arc;

use tracing::info;

use datalog_db::db::Database;
use datalog_db::server::Server;
use datalog_db::storage::rocksdb_backend::RocksDbStorage;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let args: Vec<String> = std::env::args().collect();

    let data_dir = args
        .iter()
        .position(|a| a == "--data-dir")
        .and_then(|i| args.get(i + 1))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("./datalog-data"));

    let bind_addr = args
        .iter()
        .position(|a| a == "--bind")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.as_str())
        .unwrap_or("127.0.0.1:5557");

    info!("Opening database at {:?}", data_dir);
    std::fs::create_dir_all(&data_dir)?;

    let storage = RocksDbStorage::open(&data_dir)?;
    let storage = Arc::new(storage);

    let db = Database::open(storage).await?;
    let db = Arc::new(db);

    info!("Starting server on {}", bind_addr);
    let server = Server::bind(bind_addr, db).await?;
    server.run().await?;

    Ok(())
}
