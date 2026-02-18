pub mod rocksdb_backend;

#[cfg(feature = "tikv")]
pub mod tikv;

use async_trait::async_trait;

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

#[async_trait]
pub trait StorageBackend: Send + Sync {
    async fn batch_write(&self, ops: Vec<(Vec<u8>, Vec<u8>)>) -> Result<()>;
    async fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>>;
    async fn scan(&self, start: &[u8], end: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>>;
    async fn scan_reverse(&self, start: &[u8], end: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>>;
    async fn delete(&self, key: &[u8]) -> Result<()>;
}
