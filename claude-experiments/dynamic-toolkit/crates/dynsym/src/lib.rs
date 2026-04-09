//! Interned symbols for dynamic language runtimes.
//!
//! A `Symbol` is a compile-time interned name represented as a small integer.
//! Used for method names, field names, global variable names — anywhere a
//! string identity is needed for dispatch but the actual characters don't
//! matter at runtime.
//!
//! Symbols are NOT GC values. They're plain integers that index into a
//! `SymbolTable` built at compile time. This makes them:
//! - Free to copy (u32)
//! - Free to compare (integer ==)
//! - Free to hash (already an integer)
//! - Impossible to confuse with GC string values (different type)
//!
//! # Example
//! ```
//! use dynsym::SymbolTable;
//!
//! let mut table = SymbolTable::new();
//! let init = table.intern("init");
//! let bar = table.intern("bar");
//! let init2 = table.intern("init");
//!
//! assert_eq!(init, init2);       // same string → same symbol
//! assert_ne!(init, bar);         // different string → different symbol
//! assert_eq!(table.name(init), "init");
//! assert_eq!(init.as_u32(), 0);  // first interned symbol is 0
//! assert_eq!(bar.as_u32(), 1);   // second is 1
//! ```

use std::collections::HashMap;

/// A compile-time interned name. NOT a GC value, NOT a u64.
///
/// Symbols are sequential u32 integers starting from 0. They're assigned
/// in the order strings are first interned. This makes them suitable as
/// array indices for flat dispatch tables.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct Symbol(u32);

impl Symbol {
    /// The raw integer value. Use for indexing into dispatch tables.
    #[inline(always)]
    pub const fn as_u32(self) -> u32 {
        self.0
    }

    /// Construct from a raw integer. Only use when deserializing or
    /// interfacing with code that passes symbols as integers.
    #[inline(always)]
    pub const fn from_raw(raw: u32) -> Self {
        Symbol(raw)
    }
}

impl std::fmt::Debug for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Symbol({})", self.0)
    }
}

impl std::fmt::Display for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "#{}", self.0)
    }
}

/// Maps strings to Symbols and back. Built at compile time, immutable at runtime.
///
/// Symbols are assigned sequentially: the first string interned gets Symbol(0),
/// the second gets Symbol(1), etc. Interning the same string twice returns the
/// same Symbol.
pub struct SymbolTable {
    names: Vec<String>,
    map: HashMap<String, Symbol>,
}

impl SymbolTable {
    pub fn new() -> Self {
        SymbolTable {
            names: Vec::new(),
            map: HashMap::new(),
        }
    }

    /// Intern a string, returning its Symbol. If the string was already
    /// interned, returns the existing Symbol.
    pub fn intern(&mut self, name: &str) -> Symbol {
        if let Some(&sym) = self.map.get(name) {
            return sym;
        }
        let sym = Symbol(self.names.len() as u32);
        self.map.insert(name.to_string(), sym);
        self.names.push(name.to_string());
        sym
    }

    /// Look up a Symbol's string. Panics if the Symbol wasn't created by this table.
    pub fn name(&self, sym: Symbol) -> &str {
        &self.names[sym.0 as usize]
    }

    /// Try to look up a Symbol's string. Returns None if unknown.
    pub fn try_name(&self, sym: Symbol) -> Option<&str> {
        self.names.get(sym.0 as usize).map(|s| s.as_str())
    }

    /// Look up a string's Symbol without interning it.
    pub fn lookup(&self, name: &str) -> Option<Symbol> {
        self.map.get(name).copied()
    }

    /// Total number of interned symbols. Symbols are 0..len().
    pub fn len(&self) -> usize {
        self.names.len()
    }

    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    /// Iterate all (Symbol, name) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (Symbol, &str)> {
        self.names.iter().enumerate().map(|(i, s)| (Symbol(i as u32), s.as_str()))
    }

    /// Get all interned strings as a slice (indexed by Symbol.as_u32()).
    pub fn names(&self) -> &[String] {
        &self.names
    }
}

impl Default for SymbolTable {
    fn default() -> Self {
        Self::new()
    }
}

// ── Dispatch Table ───────────────────────────────────────────────────

/// A flat lookup table keyed by Symbol.
///
/// Since Symbols are sequential integers starting at 0, the table is just
/// a Vec indexed by the symbol's raw value. Lookup is a single array index —
/// no hashing, no probing, no pointer chasing.
///
/// Entries are `u64` so they can hold NaN-boxed values (closures, etc.)
/// Use `0` or a sentinel for empty entries.
pub struct DispatchTable {
    entries: Vec<u64>,
}

/// Value indicating no entry for this symbol.
pub const DISPATCH_EMPTY: u64 = 0;

impl DispatchTable {
    /// Create an empty dispatch table.
    pub fn new() -> Self {
        DispatchTable { entries: Vec::new() }
    }

    /// Create a table pre-sized for `n` symbols.
    pub fn with_capacity(n: usize) -> Self {
        DispatchTable { entries: vec![DISPATCH_EMPTY; n] }
    }

    /// Look up a symbol. Returns None if the symbol has no entry or
    /// the entry is DISPATCH_EMPTY.
    #[inline(always)]
    pub fn get(&self, sym: Symbol) -> Option<u64> {
        let idx = sym.0 as usize;
        if idx < self.entries.len() {
            let val = self.entries[idx];
            if val != DISPATCH_EMPTY { Some(val) } else { None }
        } else {
            None
        }
    }

    /// Set an entry. Grows the table if needed.
    pub fn set(&mut self, sym: Symbol, val: u64) {
        let idx = sym.0 as usize;
        if idx >= self.entries.len() {
            self.entries.resize(idx + 1, DISPATCH_EMPTY);
        }
        self.entries[idx] = val;
    }

    /// Remove an entry (set to DISPATCH_EMPTY).
    pub fn remove(&mut self, sym: Symbol) {
        let idx = sym.0 as usize;
        if idx < self.entries.len() {
            self.entries[idx] = DISPATCH_EMPTY;
        }
    }

    /// Merge entries from another table into this one.
    /// Existing entries in `self` are NOT overwritten.
    pub fn merge_from(&mut self, other: &DispatchTable) {
        if other.entries.len() > self.entries.len() {
            self.entries.resize(other.entries.len(), DISPATCH_EMPTY);
        }
        for (i, &val) in other.entries.iter().enumerate() {
            if val != DISPATCH_EMPTY && self.entries[i] == DISPATCH_EMPTY {
                self.entries[i] = val;
            }
        }
    }

    /// Merge entries from another table, overwriting existing entries.
    pub fn merge_from_overwrite(&mut self, other: &DispatchTable) {
        if other.entries.len() > self.entries.len() {
            self.entries.resize(other.entries.len(), DISPATCH_EMPTY);
        }
        for (i, &val) in other.entries.iter().enumerate() {
            if val != DISPATCH_EMPTY {
                self.entries[i] = val;
            }
        }
    }

    /// Number of non-empty entries.
    pub fn count(&self) -> usize {
        self.entries.iter().filter(|&&v| v != DISPATCH_EMPTY).count()
    }

    /// Raw slice access for the JIT (can be loaded by base + offset).
    pub fn as_slice(&self) -> &[u64] {
        &self.entries
    }

    /// Base pointer for JIT code to index into.
    pub fn as_ptr(&self) -> *const u64 {
        self.entries.as_ptr()
    }

    /// Iterate non-empty (Symbol, value) pairs.
    pub fn iter(&self) -> impl Iterator<Item = (Symbol, u64)> + '_ {
        self.entries.iter().enumerate()
            .filter(|(_, v)| **v != DISPATCH_EMPTY)
            .map(|(i, v)| (Symbol(i as u32), *v))
    }
}

impl Default for DispatchTable {
    fn default() -> Self {
        Self::new()
    }
}

// ── Inline Cache ─────────────────────────────────────────────────────

/// A single inline cache entry for monomorphic dispatch.
///
/// Caches the result of a dynamic lookup: "for objects with this
/// class_id, the method/field is at this offset in the dispatch table,
/// and the resolved value is X."
///
/// All fields are stable integers or offsets — no GC pointers.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct InlineCacheEntry {
    /// The class identity we cached on. 0 = empty/invalid.
    /// This is a stable integer ID (e.g., the class's allocation order),
    /// NOT a GC pointer.
    pub cached_class_id: u64,
    /// The resolved value (e.g., NaN-boxed closure for method calls).
    pub cached_value: u64,
    /// The resolved code pointer (for direct calls).
    pub cached_func_ptr: u64,
}

impl InlineCacheEntry {
    pub const EMPTY: Self = InlineCacheEntry {
        cached_class_id: 0,
        cached_value: 0,
        cached_func_ptr: 0,
    };

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.cached_class_id == 0
    }
}

/// Array of inline cache entries, one per call site.
/// Allocated once per module, indexed by cache_id from the IR.
pub struct InlineCacheArray {
    entries: Vec<InlineCacheEntry>,
}

impl InlineCacheArray {
    pub fn new(num_sites: usize) -> Self {
        InlineCacheArray {
            entries: vec![InlineCacheEntry::EMPTY; num_sites],
        }
    }

    /// Base pointer for JIT code to index into.
    /// Each entry is 24 bytes (3 × u64).
    pub fn as_ptr(&self) -> *const InlineCacheEntry {
        self.entries.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut InlineCacheEntry {
        self.entries.as_mut_ptr()
    }

    /// Invalidate all cache entries (e.g., after class redefinition).
    pub fn invalidate_all(&mut self) {
        for entry in &mut self.entries {
            *entry = InlineCacheEntry::EMPTY;
        }
    }

    /// Get a specific entry.
    pub fn get(&self, cache_id: u32) -> &InlineCacheEntry {
        &self.entries[cache_id as usize]
    }

    /// Get a mutable entry (for the slow path to update).
    pub fn get_mut(&mut self, cache_id: u32) -> &mut InlineCacheEntry {
        &mut self.entries[cache_id as usize]
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symbol_interning() {
        let mut t = SymbolTable::new();
        let a = t.intern("foo");
        let b = t.intern("bar");
        let a2 = t.intern("foo");

        assert_eq!(a, a2);
        assert_ne!(a, b);
        assert_eq!(a.as_u32(), 0);
        assert_eq!(b.as_u32(), 1);
        assert_eq!(t.name(a), "foo");
        assert_eq!(t.name(b), "bar");
        assert_eq!(t.len(), 2);
    }

    #[test]
    fn symbol_lookup() {
        let mut t = SymbolTable::new();
        let s = t.intern("hello");
        assert_eq!(t.lookup("hello"), Some(s));
        assert_eq!(t.lookup("nope"), None);
    }

    #[test]
    fn dispatch_table() {
        let mut t = DispatchTable::new();
        let s0 = Symbol(0);
        let s1 = Symbol(1);
        let s5 = Symbol(5);

        assert_eq!(t.get(s0), None);

        t.set(s0, 42);
        t.set(s5, 99);

        assert_eq!(t.get(s0), Some(42));
        assert_eq!(t.get(s1), None);
        assert_eq!(t.get(s5), Some(99));
        assert_eq!(t.count(), 2);
    }

    #[test]
    fn dispatch_merge() {
        let mut parent = DispatchTable::new();
        parent.set(Symbol(0), 10);
        parent.set(Symbol(1), 20);

        let mut child = DispatchTable::new();
        child.set(Symbol(1), 30); // override parent's Symbol(1)

        child.merge_from(&parent);
        assert_eq!(child.get(Symbol(0)), Some(10)); // inherited
        assert_eq!(child.get(Symbol(1)), Some(30)); // NOT overwritten
    }

    #[test]
    fn inline_cache() {
        let mut cache = InlineCacheArray::new(4);
        assert!(cache.get(0).is_empty());

        let entry = cache.get_mut(2);
        entry.cached_class_id = 7;
        entry.cached_value = 42;
        entry.cached_func_ptr = 0xDEAD;

        assert!(!cache.get(2).is_empty());
        assert_eq!(cache.get(2).cached_class_id, 7);

        cache.invalidate_all();
        assert!(cache.get(2).is_empty());
    }
}
