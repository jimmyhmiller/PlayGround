use super::{Result, StorageBackend, StorageError};
use async_trait::async_trait;
use rocksdb::{Direction, IteratorMode, Options, WriteBatch, DB};
use std::path::Path;

pub struct RocksDbStorage {
    db: DB,
}

impl RocksDbStorage {
    pub fn open(path: &Path) -> Result<Self> {
        let mut opts = Options::default();
        opts.create_if_missing(true);
        let db = DB::open(&opts, path).map_err(|e| StorageError::Backend(e.to_string()))?;
        Ok(Self { db })
    }
}

#[async_trait]
impl StorageBackend for RocksDbStorage {
    async fn batch_write(&self, ops: Vec<(Vec<u8>, Vec<u8>)>) -> Result<()> {
        let mut batch = WriteBatch::default();
        for (key, value) in ops {
            batch.put(&key, &value);
        }
        self.db
            .write(batch)
            .map_err(|e| StorageError::Backend(e.to_string()))
    }

    async fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        self.db
            .get(key)
            .map(|opt| opt.map(|v| v.to_vec()))
            .map_err(|e| StorageError::Backend(e.to_string()))
    }

    async fn scan(&self, start: &[u8], end: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let iter = self
            .db
            .iterator(IteratorMode::From(start, Direction::Forward));
        let mut results = Vec::new();
        for item in iter {
            let (key, value) = item.map_err(|e| StorageError::Backend(e.to_string()))?;
            if key.as_ref() >= end {
                break;
            }
            results.push((key.to_vec(), value.to_vec()));
        }
        Ok(results)
    }

    async fn scan_reverse(&self, start: &[u8], end: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        // Reverse scan: iterate backward from `start`, stop before `end`
        // Caller should provide start > end
        let iter = self
            .db
            .iterator(IteratorMode::From(start, Direction::Reverse));
        let mut results = Vec::new();
        for item in iter {
            let (key, value) = item.map_err(|e| StorageError::Backend(e.to_string()))?;
            if key.as_ref() < end {
                break;
            }
            results.push((key.to_vec(), value.to_vec()));
        }
        Ok(results)
    }

    async fn delete(&self, key: &[u8]) -> Result<()> {
        self.db
            .delete(key)
            .map_err(|e| StorageError::Backend(e.to_string()))
    }
}
