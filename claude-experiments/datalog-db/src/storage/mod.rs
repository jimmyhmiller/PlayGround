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

/// Read-only operations available inside a snapshot.
pub trait ReadOps {
    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>>;
    fn scan(&self, start: &[u8], end: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>>;
    /// Iterate over entries without materializing all at once.
    /// Callback receives borrowed key/value slices. Return false to stop early.
    fn scan_foreach(
        &self,
        start: &[u8],
        end: &[u8],
        f: &mut dyn FnMut(&[u8], &[u8]) -> bool,
    ) -> Result<()>;
}

/// Read-write operations available inside a transaction.
pub trait TxnOps: ReadOps {
    fn put(&self, key: Vec<u8>, value: Vec<u8>) -> Result<()>;
    fn delete(&self, key: &[u8]) -> Result<()>;
    fn get_for_update(&self, key: &[u8], exclusive: bool) -> Result<Option<Vec<u8>>>;
}

pub type ReadCallback =
    Box<dyn FnOnce(&dyn ReadOps) -> Result<Box<dyn std::any::Any + Send>> + Send>;

pub type TxnCallback =
    Box<dyn FnOnce(&dyn TxnOps) -> Result<Box<dyn std::any::Any + Send>> + Send>;

/// Storage backend with separate read (snapshot) and read-write (transaction) paths.
pub trait StorageBackend: Send + Sync {
    /// Execute a read-only operation against a consistent snapshot.
    fn execute_read(&self, f: ReadCallback) -> Result<Box<dyn std::any::Any + Send>>;

    /// Execute a read-write transaction.
    fn execute_txn(&self, f: TxnCallback) -> Result<Box<dyn std::any::Any + Send>>;
}
