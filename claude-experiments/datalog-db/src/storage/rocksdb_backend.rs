use super::{
    BatchOverlay, GroupTxnCallback, ReadCallback, ReadOps, Result, StorageBackend, StorageError,
    TxnCallback,
};
use rocksdb::{
    checkpoint::Checkpoint, BlockBasedOptions, Cache, DBCompressionType, Direction, IteratorMode,
    Options, SnapshotWithThreadMode, WriteBatch, WriteOptions, DB,
};
use std::path::Path;
use std::sync::Arc;

/// Write-durability policy for committed transactions.
///
/// Controls the WAL behavior on each commit. None of these affect read
/// consistency or in-process ACID — only what is guaranteed to survive a
/// crash or power loss.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Durability {
    /// fsync the WAL on every commit. Safest, slowest. Survives power loss
    /// of any committed transaction.
    Sync,
    /// WAL on, no fsync. The OS may have buffered the WAL append.
    /// Survives a process crash, but a power loss can drop the tail.
    /// This is RocksDB's default and the legacy behavior of `open`.
    Buffered,
    /// WAL disabled entirely. Commits go straight to the memtable.
    /// Fastest by far, but any crash loses everything since the last
    /// memtable flush. Useful for caches, scratch databases, or tests.
    MemoryOnly,
}

impl Default for Durability {
    fn default() -> Self {
        Durability::Buffered
    }
}

/// SST file compression. `Lz4` is a balanced default; `Zstd` gives the
/// best ratio at higher CPU cost; `None` is for benchmarks or RAM-only
/// scratch databases.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Compression {
    None,
    Snappy,
    Lz4,
    Zstd,
}

impl Default for Compression {
    fn default() -> Self {
        Compression::Lz4
    }
}

impl Compression {
    fn to_rocksdb(self) -> DBCompressionType {
        match self {
            Compression::None => DBCompressionType::None,
            Compression::Snappy => DBCompressionType::Snappy,
            Compression::Lz4 => DBCompressionType::Lz4,
            Compression::Zstd => DBCompressionType::Zstd,
        }
    }
}

/// Tunable options for opening a `RocksDbStorage`.
///
/// Defaults are tuned for general use, not RocksDB's stock defaults — in
/// particular the block cache is 64 MB (vs RocksDB's 8 MB) and bloom
/// filters are enabled. Override individual fields for memory-constrained
/// or extreme workloads.
#[derive(Debug, Clone, Copy)]
pub struct StorageOptions {
    pub durability: Durability,
    /// Block cache size in bytes. RocksDB's default of 8 MB is too small
    /// for any non-trivial workload. `0` skips the override and uses
    /// RocksDB's built-in default.
    pub block_cache_bytes: usize,
    /// SST compression algorithm.
    pub compression: Compression,
    /// Bloom filter bits per key on SST data blocks. `Some(10.0)` is the
    /// standard recommendation (~1% false positive, ~10 bits per key
    /// overhead). `None` disables bloom filters. Big speedup on negative
    /// point lookups (key-not-present).
    pub bloom_filter_bits_per_key: Option<f64>,
    /// Memtable size in bytes. Larger absorbs more write bursts before
    /// flushing to SST. `0` keeps RocksDB's default.
    pub write_buffer_size: usize,
    /// How many memtables may exist concurrently before writes stall.
    /// `0` keeps RocksDB's default.
    pub max_write_buffer_number: i32,
}

impl Default for StorageOptions {
    fn default() -> Self {
        Self {
            durability: Durability::default(),
            block_cache_bytes: 64 * 1024 * 1024,
            compression: Compression::default(),
            bloom_filter_bits_per_key: Some(10.0),
            write_buffer_size: 64 * 1024 * 1024,
            max_write_buffer_number: 4,
        }
    }
}

pub struct RocksDbStorage {
    db: Arc<DB>,
    write_opts: WriteOptions,
}

impl RocksDbStorage {
    /// Open with default options.
    pub fn open(path: &Path) -> Result<Self> {
        Self::open_with(path, StorageOptions::default())
    }

    /// Open with explicit storage options.
    pub fn open_with(path: &Path, opts: StorageOptions) -> Result<Self> {
        let mut db_opts = Options::default();
        db_opts.create_if_missing(true);

        db_opts.set_compression_type(opts.compression.to_rocksdb());
        if opts.write_buffer_size > 0 {
            db_opts.set_write_buffer_size(opts.write_buffer_size);
        }
        if opts.max_write_buffer_number > 0 {
            db_opts.set_max_write_buffer_number(opts.max_write_buffer_number);
        }

        // Block-based table options (cache + bloom filter). The Cache is
        // retained inside Options' outlive struct so it outlives the DB.
        let mut block_opts = BlockBasedOptions::default();
        if opts.block_cache_bytes > 0 {
            let cache = Cache::new_lru_cache(opts.block_cache_bytes);
            block_opts.set_block_cache(&cache);
        }
        if let Some(bits) = opts.bloom_filter_bits_per_key {
            block_opts.set_bloom_filter(bits, false);
        }
        db_opts.set_block_based_table_factory(&block_opts);

        let db = DB::open(&db_opts, path)
            .map_err(|e| StorageError::Backend(e.to_string()))?;

        let mut write_opts = WriteOptions::default();
        match opts.durability {
            Durability::Sync => write_opts.set_sync(true),
            Durability::Buffered => {}
            Durability::MemoryOnly => write_opts.disable_wal(true),
        }

        Ok(Self {
            db: Arc::new(db),
            write_opts,
        })
    }
}

impl StorageBackend for RocksDbStorage {
    fn execute_read(&self, f: ReadCallback) -> Result<Box<dyn std::any::Any + Send>> {
        let snapshot = self.db.snapshot();
        let ops = RocksDbSnapshotOps { snapshot: &snapshot };
        f(&ops)
    }

    fn execute_batch(&self, f: TxnCallback) -> Result<Box<dyn std::any::Any + Send>> {
        // Snapshot serves all reads inside the callback. Writes go into
        // an in-memory overlay that's read back by the callback (so
        // read-your-own-writes works) and then committed atomically.
        let snapshot = self.db.snapshot();
        let snapshot_ops = RocksDbSnapshotOps { snapshot: &snapshot };
        let overlay = BatchOverlay::new(&snapshot_ops);

        let result = f(&overlay)?;

        // Drain pending writes into a RocksDB WriteBatch and apply
        // atomically with the configured WriteOptions (which encode the
        // Durability policy chosen at open time).
        let mut batch = WriteBatch::default();
        for (key, value) in overlay.into_writes() {
            match value {
                Some(v) => batch.put(&key, &v),
                None => batch.delete(&key),
            }
        }
        if !batch.is_empty() {
            self.db
                .write_opt(batch, &self.write_opts)
                .map_err(|e| StorageError::Backend(e.to_string()))?;
        }

        Ok(result)
    }

    fn execute_group_batch(
        &self,
        callbacks: Vec<GroupTxnCallback>,
    ) -> Result<Vec<Result<Box<dyn std::any::Any + Send>>>> {
        // One snapshot, one overlay shared across all callbacks. Each
        // callback runs with its own checkpoint boundary so a failure
        // affects only that callback's writes.
        let snapshot = self.db.snapshot();
        let snapshot_ops = RocksDbSnapshotOps { snapshot: &snapshot };
        let overlay = BatchOverlay::new(&snapshot_ops);

        let mut results: Vec<Result<Box<dyn std::any::Any + Send>>> =
            Vec::with_capacity(callbacks.len());
        for cb in callbacks {
            let cp = overlay.checkpoint();
            match cb(&overlay) {
                Ok(r) => results.push(Ok(r)),
                Err(e) => {
                    overlay.rollback_to(cp);
                    results.push(Err(e));
                }
            }
        }

        let mut batch = WriteBatch::default();
        for (key, value) in overlay.into_writes() {
            match value {
                Some(v) => batch.put(&key, &v),
                None => batch.delete(&key),
            }
        }
        if !batch.is_empty() {
            self.db
                .write_opt(batch, &self.write_opts)
                .map_err(|e| StorageError::Backend(e.to_string()))?;
        }

        Ok(results)
    }

    fn checkpoint(&self, path: &std::path::Path) -> Result<()> {
        let cp = Checkpoint::new(self.db.as_ref())
            .map_err(|e| StorageError::Backend(format!("checkpoint init: {}", e)))?;
        cp.create_checkpoint(path)
            .map_err(|e| StorageError::Backend(format!("checkpoint create: {}", e)))?;
        Ok(())
    }
}

// -- Snapshot (read-only) --

struct RocksDbSnapshotOps<'a> {
    snapshot: &'a SnapshotWithThreadMode<'a, DB>,
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

// Note: the dedicated transaction-bound `TxnOps` impl has been removed.
// Writes now flow through `BatchOverlay` in `storage::mod` — the snapshot
// reads happen via `RocksDbSnapshotOps`, and the overlay's accumulated
// writes are applied as one atomic `WriteBatch` in `execute_batch`.
