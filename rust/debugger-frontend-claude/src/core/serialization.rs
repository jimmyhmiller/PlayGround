use bincode::{config::standard, Decode, Encode};
use anyhow::Result;

/// Trait for binary serialization of debugging messages
pub trait BinarySerialize {
    fn to_binary(&self) -> Result<Vec<u8>>;
    fn from_binary(data: &[u8]) -> Result<Self>
    where
        Self: Sized;
}

impl<T: Encode + Decode<()>> BinarySerialize for T {
    fn to_binary(&self) -> Result<Vec<u8>> {
        bincode::encode_to_vec(self, standard())
            .map_err(|e| anyhow::anyhow!("Serialization error: {}", e))
    }

    fn from_binary(data: &[u8]) -> Result<T> {
        let (decoded, _) = bincode::decode_from_slice(data, standard())
            .map_err(|e| anyhow::anyhow!("Deserialization error: {}", e))?;
        Ok(decoded)
    }
}

/// Utility functions for memory operations
pub fn convert_to_u64_array(input: &[u8; 512]) -> Vec<u64> {
    input
        .chunks(8)
        .map(|chunk| {
            chunk
                .iter()
                .enumerate()
                .fold(0, |acc, (i, &byte)| acc | (byte as u64) << (8 * i))
        })
        .collect()
}