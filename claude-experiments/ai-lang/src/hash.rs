//! Content hashes.
//!
//! A `Hash` is the Blake3 digest of a canonical AST's binary encoding
//! (see `codec`). Two definitions with the same `Hash` are byte-identical
//! in canonical form, and therefore the same definition.

use core::fmt;

/// 32-byte Blake3 content hash.
#[derive(Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Hash(pub [u8; 32]);

impl Hash {
    pub const SIZE: usize = 32;

    /// Hash the canonical-encoded bytes of a definition or expression.
    pub fn of_bytes(bytes: &[u8]) -> Self {
        Hash(*blake3::hash(bytes).as_bytes())
    }

    pub fn as_bytes(&self) -> &[u8; 32] {
        &self.0
    }

    /// Lower-case hex representation. Used in error messages and as the
    /// disk filename for the canonical store.
    pub fn to_hex(&self) -> String {
        let mut s = String::with_capacity(64);
        for b in &self.0 {
            s.push(hex_nibble(b >> 4));
            s.push(hex_nibble(b & 0xf));
        }
        s
    }
}

impl fmt::Debug for Hash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Short form: first 8 hex chars + "..." — full hash on demand via to_hex.
        let h = self.to_hex();
        write!(f, "Hash({}…)", &h[..8])
    }
}

impl fmt::Display for Hash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

fn hex_nibble(n: u8) -> char {
    match n {
        0..=9 => (b'0' + n) as char,
        10..=15 => (b'a' + (n - 10)) as char,
        _ => unreachable!("nibble out of range"),
    }
}
