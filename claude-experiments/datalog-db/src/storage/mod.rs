pub mod rocksdb_backend;

pub use rocksdb_backend::{Compression, Durability, StorageOptions};

#[cfg(feature = "tikv")]
pub mod tikv;

use std::cell::RefCell;
use std::collections::BTreeMap;

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
///
/// The batch backend implements this trait with `get_for_update`
/// degraded to a plain `get`. Concurrency is provided by a higher-level
/// `Mutex` on the `Database` rather than by row locks in RocksDB.
pub trait TxnOps: ReadOps {
    fn put(&self, key: Vec<u8>, value: Vec<u8>) -> Result<()>;
    fn delete(&self, key: &[u8]) -> Result<()>;
    fn get_for_update(&self, key: &[u8], exclusive: bool) -> Result<Option<Vec<u8>>>;
}

pub type ReadCallback =
    Box<dyn FnOnce(&dyn ReadOps) -> Result<Box<dyn std::any::Any + Send>> + Send>;

pub type TxnCallback =
    Box<dyn FnOnce(&dyn TxnOps) -> Result<Box<dyn std::any::Any + Send>> + Send>;

/// Callback for one user transaction within a group commit. Returns
/// either a successful result (which carries that tx's writes into the
/// shared batch) or an error (in which case the tx's writes are rolled
/// back and only that tx fails — its siblings in the group still commit).
pub type GroupTxnCallback =
    Box<dyn FnOnce(&dyn TxnOps) -> Result<Box<dyn std::any::Any + Send>> + Send>;

/// Storage backend with separate read (snapshot) and read-write (batch) paths.
pub trait StorageBackend: Send + Sync {
    /// Execute a read-only operation against a consistent snapshot.
    fn execute_read(&self, f: ReadCallback) -> Result<Box<dyn std::any::Any + Send>>;

    /// Execute a read-write batch.
    ///
    /// The callback runs against a snapshot of the database with an
    /// in-memory write overlay. All puts/deletes accumulate in the
    /// overlay; reads (including `scan` / `scan_foreach`) see the
    /// overlay union'd with the snapshot, preserving read-your-own-writes
    /// semantics within the callback. On a successful return, all
    /// accumulated writes are applied as a single atomic `WriteBatch`.
    ///
    /// Higher-level coordination (tx_id allocation, serialization) is
    /// the caller's responsibility — typically a `Mutex` on the
    /// `Database`.
    fn execute_batch(&self, f: TxnCallback) -> Result<Box<dyn std::any::Any + Send>>;

    /// Execute multiple user transactions as one group commit.
    ///
    /// Each callback runs in sequence against a shared overlay layered
    /// on a single snapshot. Earlier callbacks' successful writes are
    /// visible to later callbacks (read-your-prior-writes within the
    /// group). A failing callback rolls back to its checkpoint —
    /// siblings keep going. On return, the union of all successful
    /// callbacks' writes is applied as a single atomic WriteBatch
    /// (one fsync, one WAL append).
    ///
    /// Returns a result per input callback in the same order. A
    /// backend-level failure of the batch apply itself returns an
    /// outer `Err` (in which case no callbacks have visibly committed).
    fn execute_group_batch(
        &self,
        callbacks: Vec<GroupTxnCallback>,
    ) -> Result<Vec<Result<Box<dyn std::any::Any + Send>>>>;

    /// Produce a point-in-time on-disk checkpoint of the database at
    /// `path`. The default impl returns an "unsupported" error; backends
    /// that can checkpoint (RocksDB) override this. `path` must not yet
    /// exist; it will be created. For RocksDB the checkpoint is hard-link
    /// based, so `path` must be on the same filesystem as the live DB.
    fn checkpoint(&self, _path: &std::path::Path) -> Result<()> {
        Err(StorageError::Backend(
            "checkpoint not supported by this storage backend".into(),
        ))
    }
}

// --- Pending write overlay used by all batch backends ---

/// A staged write: `Some(value)` is a put, `None` is a delete.
type PendingWrite = Option<Vec<u8>>;

/// Wraps a read snapshot with an in-memory overlay of pending writes,
/// so that reads inside a batch callback see writes the callback has
/// already issued ("read your own writes" within one batch).
///
/// Used by `RocksDbStorage::execute_batch`; would be reused by any other
/// backend that wants to provide WriteBatch semantics.
pub struct BatchOverlay<'a> {
    snapshot: &'a dyn ReadOps,
    /// Sorted map so `scan` returns entries in key order. Overlay
    /// entries override snapshot entries on identical keys; a `None`
    /// value means the key has been deleted in the overlay.
    pending: RefCell<BTreeMap<Vec<u8>, PendingWrite>>,
}

impl<'a> BatchOverlay<'a> {
    pub fn new(snapshot: &'a dyn ReadOps) -> Self {
        Self {
            snapshot,
            pending: RefCell::new(BTreeMap::new()),
        }
    }

    /// Drain the staged writes for batch application by the backend.
    pub fn into_writes(self) -> Vec<(Vec<u8>, PendingWrite)> {
        self.pending.into_inner().into_iter().collect()
    }

    /// Snapshot the current pending writes. Pair with `rollback_to` to
    /// undo any writes made between checkpoint and rollback. Used by
    /// `execute_group_batch` to give each user transaction in a group
    /// its own atomic commit boundary inside one shared WriteBatch.
    pub fn checkpoint(&self) -> OverlayCheckpoint {
        OverlayCheckpoint {
            pending: self.pending.borrow().clone(),
        }
    }

    /// Restore the pending writes to the state captured in `checkpoint`.
    /// Any puts/deletes issued after the checkpoint are discarded.
    pub fn rollback_to(&self, checkpoint: OverlayCheckpoint) {
        *self.pending.borrow_mut() = checkpoint.pending;
    }
}

/// Snapshot of a `BatchOverlay`'s pending writes. Opaque to callers
/// outside the storage layer.
pub struct OverlayCheckpoint {
    pending: BTreeMap<Vec<u8>, PendingWrite>,
}

impl ReadOps for BatchOverlay<'_> {
    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        if let Some(pending) = self.pending.borrow().get(key) {
            return Ok(pending.clone());
        }
        self.snapshot.get(key)
    }

    fn scan(&self, start: &[u8], end: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let mut merged: BTreeMap<Vec<u8>, Vec<u8>> = BTreeMap::new();

        // Pull snapshot entries in range.
        for (k, v) in self.snapshot.scan(start, end)? {
            merged.insert(k, v);
        }

        // Apply overlay entries in range — put overrides, delete removes.
        for (k, v_opt) in self.pending.borrow().range(start.to_vec()..end.to_vec()) {
            match v_opt {
                Some(v) => {
                    merged.insert(k.clone(), v.clone());
                }
                None => {
                    merged.remove(k);
                }
            }
        }

        Ok(merged.into_iter().collect())
    }

    fn scan_foreach(
        &self,
        start: &[u8],
        end: &[u8],
        f: &mut dyn FnMut(&[u8], &[u8]) -> bool,
    ) -> Result<()> {
        // Simpler to materialize and walk than to do a true merge-sort
        // between two iterators while honoring the overlay semantics.
        // Range sizes here are small (per-entity, per-attribute) so
        // this is fine in practice.
        for (k, v) in self.scan(start, end)? {
            if !f(&k, &v) {
                break;
            }
        }
        Ok(())
    }
}

impl TxnOps for BatchOverlay<'_> {
    fn put(&self, key: Vec<u8>, value: Vec<u8>) -> Result<()> {
        self.pending.borrow_mut().insert(key, Some(value));
        Ok(())
    }

    fn delete(&self, key: &[u8]) -> Result<()> {
        self.pending.borrow_mut().insert(key.to_vec(), None);
        Ok(())
    }

    /// In the batch backend, callers serialize writers themselves via a
    /// `Mutex` on the `Database`. There is no row-level lock manager to
    /// engage, so this degrades to a plain `get` against the overlay.
    fn get_for_update(&self, key: &[u8], _exclusive: bool) -> Result<Option<Vec<u8>>> {
        self.get(key)
    }
}
