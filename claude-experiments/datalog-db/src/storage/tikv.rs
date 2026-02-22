use super::{Result, StorageBackend, StorageError, TxnCallback, TxnOps};
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

struct TiKvTxnOps<'a> {
    txn: RefCell<tikv_client::Transaction>,
    prefix: Vec<u8>,
    runtime: &'a tokio::runtime::Runtime,
}

impl TiKvTxnOps<'_> {
    fn prefixed_key(&self, key: &[u8]) -> Vec<u8> {
        let mut k = self.prefix.clone();
        k.extend_from_slice(key);
        k
    }

    fn strip_prefix_vec(&self, key: Vec<u8>) -> Vec<u8> {
        key[self.prefix.len()..].to_vec()
    }
}

impl TxnOps for TiKvTxnOps<'_> {
    fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let k = self.prefixed_key(key);
        self.runtime
            .block_on(self.txn.borrow_mut().get(k))
            .map_err(|e| StorageError::Backend(e.to_string()))
    }

    fn scan(&self, start: &[u8], end: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let start = self.prefixed_key(start);
        let end = self.prefixed_key(end);
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
                (self.strip_prefix_vec(key), value)
            })
            .collect())
    }

    fn put(&self, key: Vec<u8>, value: Vec<u8>) -> Result<()> {
        let k = self.prefixed_key(&key);
        self.runtime
            .block_on(self.txn.borrow_mut().put(k, value))
            .map_err(|e| StorageError::Backend(e.to_string()))
    }

    fn get_for_update(&self, key: &[u8], _exclusive: bool) -> Result<Option<Vec<u8>>> {
        let k = self.prefixed_key(key);
        self.runtime
            .block_on(self.txn.borrow_mut().get_for_update(k))
            .map_err(|e| StorageError::Backend(e.to_string()))
    }
}
