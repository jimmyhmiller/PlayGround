use super::{ReadCallback, ReadOps, Result, StorageBackend, StorageError, TxnCallback, TxnOps};
use std::cell::RefCell;
use tikv_client::{KvPair, TransactionClient};

pub struct TiKvStorage {
    client: TransactionClient,
    /// Key prefix to namespace this database's data in the TiKV cluster.
    prefix: Vec<u8>,
    runtime: tokio::runtime::Runtime,
}

impl TiKvStorage {
    pub fn connect(pd_endpoints: Vec<String>, prefix: &str) -> Result<Self> {
        let runtime = tokio::runtime::Runtime::new()
            .map_err(|e| StorageError::Backend(e.to_string()))?;
        let client = runtime
            .block_on(TransactionClient::new(pd_endpoints))
            .map_err(|e| StorageError::Backend(e.to_string()))?;
        Ok(Self {
            client,
            prefix: prefix.as_bytes().to_vec(),
            runtime,
        })
    }
}

impl StorageBackend for TiKvStorage {
    fn execute_read(&self, f: ReadCallback) -> Result<Box<dyn std::any::Any + Send>> {
        let snapshot = self
            .runtime
            .block_on(self.client.snapshot(
                self.runtime
                    .block_on(self.client.current_timestamp())
                    .map_err(|e| StorageError::Backend(e.to_string()))?,
                tikv_client::BoundRange::from(..),
            ))
            .map_err(|e| StorageError::Backend(e.to_string()))?;

        let ops = TiKvSnapshotOps {
            snapshot: RefCell::new(snapshot),
            prefix: self.prefix.clone(),
            runtime: &self.runtime,
        };

        f(&ops)
    }

    fn execute_txn(&self, f: TxnCallback) -> Result<Box<dyn std::any::Any + Send>> {
        let txn = self
            .runtime
            .block_on(self.client.begin_pessimistic())
            .map_err(|e| StorageError::Backend(e.to_string()))?;

        let ops = TiKvTxnOps {
            txn: RefCell::new(txn),
            prefix: self.prefix.clone(),
            runtime: &self.runtime,
        };

        let result = f(&ops)?;

        let txn = ops.txn.into_inner();
        self.runtime
            .block_on(txn.commit())
            .map_err(|e| StorageError::Backend(e.to_string()))?;

        Ok(result)
    }
}

// -- Helpers --

fn prefixed_key(prefix: &[u8], key: &[u8]) -> Vec<u8> {
    let mut k = prefix.to_vec();
    k.extend_from_slice(key);
    k
}

fn strip_prefix(prefix: &[u8], key: Vec<u8>) -> Vec<u8> {
    key[prefix.len()..].to_vec()
}

// -- Snapshot (read-only) --

struct TiKvSnapshotOps<'a> {
    snapshot: RefCell<tikv_client::Transaction>,
    prefix: Vec<u8>,
    runtime: &'a tokio::runtime::Runtime,
}

impl ReadOps for TiKvSnapshotOps<'_> {
    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let k = prefixed_key(&self.prefix, key);
        self.runtime
            .block_on(self.snapshot.borrow_mut().get(k))
            .map_err(|e| StorageError::Backend(e.to_string()))
    }

    fn scan(&self, start: &[u8], end: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let start = prefixed_key(&self.prefix, start);
        let end = prefixed_key(&self.prefix, end);
        let pairs: Vec<KvPair> = self
            .runtime
            .block_on(self.snapshot.borrow_mut().scan(start..end, u32::MAX))
            .map_err(|e| StorageError::Backend(e.to_string()))?
            .collect();
        Ok(pairs
            .into_iter()
            .map(|kv| {
                let key: Vec<u8> = kv.0.into();
                let value: Vec<u8> = kv.1;
                (strip_prefix(&self.prefix, key), value)
            })
            .collect())
    }
}

// -- Transaction (read-write) --

struct TiKvTxnOps<'a> {
    txn: RefCell<tikv_client::Transaction>,
    prefix: Vec<u8>,
    runtime: &'a tokio::runtime::Runtime,
}

impl ReadOps for TiKvTxnOps<'_> {
    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let k = prefixed_key(&self.prefix, key);
        self.runtime
            .block_on(self.txn.borrow_mut().get(k))
            .map_err(|e| StorageError::Backend(e.to_string()))
    }

    fn scan(&self, start: &[u8], end: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let start = prefixed_key(&self.prefix, start);
        let end = prefixed_key(&self.prefix, end);
        let pairs: Vec<KvPair> = self
            .runtime
            .block_on(self.txn.borrow_mut().scan(start..end, u32::MAX))
            .map_err(|e| StorageError::Backend(e.to_string()))?
            .collect();
        Ok(pairs
            .into_iter()
            .map(|kv| {
                let key: Vec<u8> = kv.0.into();
                let value: Vec<u8> = kv.1;
                (strip_prefix(&self.prefix, key), value)
            })
            .collect())
    }
}

impl TxnOps for TiKvTxnOps<'_> {
    fn put(&self, key: Vec<u8>, value: Vec<u8>) -> Result<()> {
        let k = prefixed_key(&self.prefix, &key);
        self.runtime
            .block_on(self.txn.borrow_mut().put(k, value))
            .map_err(|e| StorageError::Backend(e.to_string()))
    }

    fn get_for_update(&self, key: &[u8], _exclusive: bool) -> Result<Option<Vec<u8>>> {
        let k = prefixed_key(&self.prefix, key);
        self.runtime
            .block_on(self.txn.borrow_mut().get_for_update(k))
            .map_err(|e| StorageError::Backend(e.to_string()))
    }
}
