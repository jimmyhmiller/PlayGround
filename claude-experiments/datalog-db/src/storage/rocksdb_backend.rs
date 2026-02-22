use super::{ReadCallback, ReadOps, Result, StorageBackend, StorageError, TxnCallback, TxnOps};
use rocksdb::{Direction, IteratorMode, Options, SnapshotWithThreadMode, TransactionDB, TransactionDBOptions};
use std::path::Path;
use std::sync::Arc;

pub struct RocksDbStorage {
    db: Arc<TransactionDB>,
}

impl RocksDbStorage {
    pub fn open(path: &Path) -> Result<Self> {
        let mut opts = Options::default();
        opts.create_if_missing(true);
        let txn_db_opts = TransactionDBOptions::default();
        let db = TransactionDB::open(&opts, &txn_db_opts, path)
            .map_err(|e| StorageError::Backend(e.to_string()))?;
        Ok(Self { db: Arc::new(db) })
    }
}

impl StorageBackend for RocksDbStorage {
    fn execute_read(&self, f: ReadCallback) -> Result<Box<dyn std::any::Any + Send>> {
        let snapshot = self.db.snapshot();
        let ops = RocksDbSnapshotOps { snapshot: &snapshot };
        f(&ops)
    }

    fn execute_txn(&self, f: TxnCallback) -> Result<Box<dyn std::any::Any + Send>> {
        let txn = self.db.transaction();
        let ops = RocksDbTxnOps { txn: &txn };
        let result = f(&ops)?;
        txn.commit()
            .map_err(|e| StorageError::Backend(e.to_string()))?;
        Ok(result)
    }
}

// -- Snapshot (read-only) --

struct RocksDbSnapshotOps<'a> {
    snapshot: &'a SnapshotWithThreadMode<'a, TransactionDB>,
}

impl ReadOps for RocksDbSnapshotOps<'_> {
    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        self.snapshot
            .get(key)
            .map(|opt| opt.map(|v| v.to_vec()))
            .map_err(|e| StorageError::Backend(e.to_string()))
    }

    fn scan(&self, start: &[u8], end: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let iter = self
            .snapshot
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

    fn scan_foreach(
        &self,
        start: &[u8],
        end: &[u8],
        f: &mut dyn FnMut(&[u8], &[u8]) -> bool,
    ) -> Result<()> {
        let iter = self
            .snapshot
            .iterator(IteratorMode::From(start, Direction::Forward));
        for item in iter {
            let (key, value) = item.map_err(|e| StorageError::Backend(e.to_string()))?;
            if key.as_ref() >= end {
                break;
            }
            if !f(&key, &value) {
                break;
            }
        }
        Ok(())
    }
}

// -- Transaction (read-write) --

struct RocksDbTxnOps<'a> {
    txn: &'a rocksdb::Transaction<'a, TransactionDB>,
}

impl ReadOps for RocksDbTxnOps<'_> {
    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        self.txn
            .get(key)
            .map(|opt| opt.map(|v| v.to_vec()))
            .map_err(|e| StorageError::Backend(e.to_string()))
    }

    fn scan(&self, start: &[u8], end: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let iter = self
            .txn
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

    fn scan_foreach(
        &self,
        start: &[u8],
        end: &[u8],
        f: &mut dyn FnMut(&[u8], &[u8]) -> bool,
    ) -> Result<()> {
        let iter = self
            .txn
            .iterator(IteratorMode::From(start, Direction::Forward));
        for item in iter {
            let (key, value) = item.map_err(|e| StorageError::Backend(e.to_string()))?;
            if key.as_ref() >= end {
                break;
            }
            if !f(&key, &value) {
                break;
            }
        }
        Ok(())
    }
}

impl TxnOps for RocksDbTxnOps<'_> {
    fn put(&self, key: Vec<u8>, value: Vec<u8>) -> Result<()> {
        self.txn
            .put(&key, &value)
            .map_err(|e| StorageError::Backend(e.to_string()))
    }

    fn delete(&self, key: &[u8]) -> Result<()> {
        self.txn
            .delete(key)
            .map_err(|e| StorageError::Backend(e.to_string()))
    }

    fn get_for_update(&self, key: &[u8], exclusive: bool) -> Result<Option<Vec<u8>>> {
        self.txn
            .get_for_update(key, exclusive)
            .map(|opt| opt.map(|v| v.to_vec()))
            .map_err(|e| StorageError::Backend(e.to_string()))
    }
}
