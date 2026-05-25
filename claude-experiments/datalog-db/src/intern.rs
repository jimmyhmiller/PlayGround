//! Attribute name interning.
//!
//! Every index key holds an attribute identifier. Before interning,
//! that identifier was the full attribute name inlined as
//! `[u16 len][bytes]` — typically 10-30 bytes per key, repeated across
//! 5+ keys per datom (EAVT + AEVT + AVET + CURRENT_AEVT +
//! CURRENT_AVET, plus VAET for refs).
//!
//! With interning, the identifier is a fixed 4-byte `AttrId` (`u32`).
//! A meta-key mapping (`attr:<name>` → `u32` big-endian) records the
//! assignment; the in-memory `AttrInterner` mirrors that mapping for
//! fast lookup.
//!
//! Allocation happens on first write of a new attribute. The
//! `attr:<name>` put goes into the same `WriteBatch` as the datoms
//! that reference it — atomically, so a crash never leaves a datom
//! referencing an undefined ID.
//!
//! Lookups during reads return `Option<AttrId>`: `None` means the
//! attribute has never been seen, in which case scans for it return
//! nothing (correctly — no datoms use it).
//!
//! `AttrId::INVALID == 0` is reserved as a sentinel: allocations
//! start at 1. Scanning a prefix built around id 0 matches no real
//! data, which lets callers handle "missing attribute" by passing 0
//! through instead of branching on `Option`.

use std::any::Any;
use std::collections::HashMap;

use byteorder::{BigEndian, ByteOrder};
use parking_lot::RwLock;

use crate::datom::Datom;
use crate::index::{self, DecodedDatom};
use crate::storage::{StorageBackend, StorageError, TxnOps};

/// Storage-layer attribute identifier. Allocated on first write of a
/// new attribute name; persisted to a meta-key table; small (`u32`).
pub type AttrId = u32;

/// Reserved sentinel — no real attribute ever has this id.
/// Lets prefix-building code that can't find a name return this and
/// have the resulting scan match nothing.
pub const INVALID_ATTR_ID: AttrId = 0;

const ATTR_META_PREFIX: &str = "attr:";

/// Build the meta-key under which `attr:<name>` is stored.
fn attr_meta_key(name: &str) -> Vec<u8> {
    index::meta_key(&format!("{}{}", ATTR_META_PREFIX, name))
}

/// Two-way mapping between attribute names and `AttrId`s. Persisted
/// behind `attr:<name>` meta keys; mirrored in memory for fast
/// lookup.
pub struct AttrInterner {
    inner: RwLock<InternerInner>,
}

struct InternerInner {
    forward: HashMap<String, AttrId>,
    reverse: HashMap<AttrId, String>,
    /// Next id to hand out on a fresh allocation.
    next_id: AttrId,
}

impl AttrInterner {
    /// Construct an empty interner. Allocations begin at id 1.
    pub fn new() -> Self {
        Self {
            inner: RwLock::new(InternerInner {
                forward: HashMap::new(),
                reverse: HashMap::new(),
                next_id: 1,
            }),
        }
    }

    /// Populate the in-memory state by scanning the `attr:` meta-key
    /// range. Called once at `Database::open`.
    pub fn load_from_storage(
        &self,
        storage: &dyn StorageBackend,
    ) -> std::result::Result<(), StorageError> {
        let result = storage.execute_read(Box::new(|snap| {
            let prefix = index::meta_key(ATTR_META_PREFIX);
            let end = index::prefix_end(&prefix);
            let entries = snap.scan(&prefix, &end)?;
            // (name, id) pairs. The key layout is
            // [META_PREFIX][b"attr:"][name_bytes], so the name begins
            // at offset 1 + len("attr:") = 6.
            let prefix_len = prefix.len();
            let mut pairs: Vec<(String, AttrId)> = Vec::with_capacity(entries.len());
            for (k, v) in entries {
                if k.len() < prefix_len || v.len() != 4 {
                    continue;
                }
                let name_bytes = &k[prefix_len..];
                let name = match std::str::from_utf8(name_bytes) {
                    Ok(s) => s.to_string(),
                    Err(_) => continue,
                };
                let id = BigEndian::read_u32(&v);
                pairs.push((name, id));
            }
            Ok(Box::new(pairs) as Box<dyn Any + Send>)
        }))?;

        let pairs = *result
            .downcast::<Vec<(String, AttrId)>>()
            .expect("attr-load result type");

        let mut inner = self.inner.write();
        for (name, id) in pairs {
            inner.forward.insert(name.clone(), id);
            inner.reverse.insert(id, name);
            if id >= inner.next_id {
                inner.next_id = id + 1;
            }
        }
        Ok(())
    }

    /// Read-only lookup. Returns `None` if the attribute has never been
    /// allocated — callers building scan prefixes should treat this as
    /// "no matching data" rather than as an error.
    pub fn lookup(&self, name: &str) -> Option<AttrId> {
        self.inner.read().forward.get(name).copied()
    }

    /// Convenience for prefix builders: returns the id if known,
    /// otherwise the reserved invalid id (which matches no real data
    /// when scanned).
    pub fn lookup_or_invalid(&self, name: &str) -> AttrId {
        self.lookup(name).unwrap_or(INVALID_ATTR_ID)
    }

    /// Reverse lookup. Returns the attribute name for a given id, or
    /// `None` if the id has never been allocated.
    pub fn name_of(&self, id: AttrId) -> Option<String> {
        self.inner.read().reverse.get(&id).cloned()
    }

    /// Intern an attribute name, allocating a new id if necessary.
    ///
    /// Must be called with the `Database` write lock held — the caller
    /// is responsible for ensuring no two threads allocate IDs
    /// concurrently. The newly-allocated `attr:<name>` row is written
    /// into `txn`, which the caller will commit together with the
    /// datoms that reference the id.
    pub fn intern(
        &self,
        txn: &dyn TxnOps,
        name: &str,
    ) -> std::result::Result<AttrId, StorageError> {
        // Fast path under the read lock first.
        if let Some(id) = self.lookup(name) {
            return Ok(id);
        }

        let mut inner = self.inner.write();
        // Re-check now that we hold the write lock — another
        // intern call on the same name could have lost the race.
        if let Some(&id) = inner.forward.get(name) {
            return Ok(id);
        }

        let id = inner.next_id;
        inner.next_id = inner
            .next_id
            .checked_add(1)
            .expect("AttrId space exhausted");
        inner.forward.insert(name.to_string(), id);
        inner.reverse.insert(id, name.to_string());

        // Persist the mapping into the same write batch as the data
        // that will use it. The `BatchOverlay` makes this visible to
        // subsequent reads in the same transaction via
        // read-your-own-writes.
        let key = attr_meta_key(name);
        let mut value = [0u8; 4];
        BigEndian::write_u32(&mut value, id);
        txn.put(key, value.to_vec())?;

        Ok(id)
    }

    /// Number of attributes currently interned. Useful as a
    /// diagnostic and for verifying that repeated writes of the same
    /// attribute don't allocate fresh IDs.
    pub fn len(&self) -> usize {
        self.inner.read().forward.len()
    }

    /// Whether anything has been interned yet.
    pub fn is_empty(&self) -> bool {
        self.inner.read().forward.is_empty()
    }

    /// Resolve a freshly-decoded storage datom into a user-facing
    /// `Datom` with the attribute name materialized from the in-memory
    /// table. Returns `None` if the id has never been allocated, which
    /// can only happen if the persisted attr table is out of sync with
    /// the index data — a sign of corruption.
    pub fn resolve(&self, decoded: DecodedDatom) -> Option<Datom> {
        let attribute = self.name_of(decoded.attr_id)?;
        Some(Datom {
            entity: decoded.entity,
            attribute,
            value: decoded.value,
            tx: decoded.tx,
            added: decoded.added,
        })
    }
}

impl Default for AttrInterner {
    fn default() -> Self {
        Self::new()
    }
}
