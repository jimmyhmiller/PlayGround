use super::{Result, StorageBackend, StorageError};
use async_trait::async_trait;
use tikv_client::RawClient;

pub struct TiKvStorage {
    client: RawClient,
    /// Key prefix to namespace this database's data in the TiKV cluster.
    prefix: Vec<u8>,
}

impl TiKvStorage {
    pub async fn connect(pd_endpoints: Vec<String>, prefix: &str) -> Result<Self> {
        let client = RawClient::new(pd_endpoints)
            .await
            .map_err(|e| StorageError::Backend(e.to_string()))?;
        Ok(Self {
            client,
            prefix: prefix.as_bytes().to_vec(),
        })
    }

    fn prefixed_key(&self, key: &[u8]) -> Vec<u8> {
        let mut k = self.prefix.clone();
        k.extend_from_slice(key);
        k
    }

    fn strip_prefix<'a>(&self, key: &'a [u8]) -> &'a [u8] {
        &key[self.prefix.len()..]
    }
}

#[async_trait]
impl StorageBackend for TiKvStorage {
    async fn batch_write(&self, ops: Vec<(Vec<u8>, Vec<u8>)>) -> Result<()> {
        let prefixed: Vec<_> = ops
            .into_iter()
            .map(|(k, v)| (self.prefixed_key(&k), v))
            .collect();
        self.client
            .batch_put(prefixed)
            .await
            .map_err(|e| StorageError::Backend(e.to_string()))
    }

    async fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let k = self.prefixed_key(key);
        self.client
            .get(k)
            .await
            .map_err(|e| StorageError::Backend(e.to_string()))
    }

    async fn scan(&self, start: &[u8], end: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        let start = self.prefixed_key(start);
        let end = self.prefixed_key(end);
        let pairs = self
            .client
            .scan(start..end, u32::MAX)
            .await
            .map_err(|e| StorageError::Backend(e.to_string()))?;
        Ok(pairs
            .into_iter()
            .map(|kv| {
                let key: Vec<u8> = kv.key().into();
                let value: Vec<u8> = kv.value().into();
                (self.strip_prefix(&key).to_vec(), value)
            })
            .collect())
    }

    async fn scan_reverse(&self, start: &[u8], end: &[u8]) -> Result<Vec<(Vec<u8>, Vec<u8>)>> {
        // TiKV RawClient doesn't have native reverse scan,
        // so we scan forward and reverse the results.
        let start_key = self.prefixed_key(end);
        let end_key = self.prefixed_key(start);
        let pairs = self
            .client
            .scan(start_key..end_key, u32::MAX)
            .await
            .map_err(|e| StorageError::Backend(e.to_string()))?;
        let mut result: Vec<_> = pairs
            .into_iter()
            .map(|kv| {
                let key: Vec<u8> = kv.key().into();
                let value: Vec<u8> = kv.value().into();
                (self.strip_prefix(&key).to_vec(), value)
            })
            .collect();
        result.reverse();
        Ok(result)
    }

    async fn delete(&self, key: &[u8]) -> Result<()> {
        let k = self.prefixed_key(key);
        self.client
            .delete(k)
            .await
            .map_err(|e| StorageError::Backend(e.to_string()))
    }
}
