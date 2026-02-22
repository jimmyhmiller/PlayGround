pub mod rocksdb_backend;

#[cfg(feature = "tikv")]
pub mod tikv;

#[derive(Debug, thiserror::Error)]
pub enum StorageError {
    #[error("Storage I/O error: {0}")]
    Io(String),
    #[error("Key not found")]
    NotFound,
    #[error("Storage backend error: {0}")]
    Backend(String),
}

pub type Result<T> = std::result::Result<T, StorageError>;

/// Synchronous read/write operations available inside a transaction.
pub trait TxnOps {
    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>>;
    fn scan(&self, start: &[u8], end: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>>;
    fn put(&self, key: Vec<u8>, value: Vec<u8>) -> Result<()>;
    fn get_for_update(&self, key: &[u8], exclusive: bool) -> Result<Option<Vec<u8>>>;
}

pub type TxnCallback =
    Box<dyn FnOnce(&dyn TxnOps) -> Result<Box<dyn std::any::Any + Send>> + Send>;

/// Storage backend. All operations run inside transactions via `execute_txn`.
pub trait StorageBackend: Send + Sync {
    fn execute_txn(&self, f: TxnCallback) -> Result<Box<dyn std::any::Any + Send>>;
}
