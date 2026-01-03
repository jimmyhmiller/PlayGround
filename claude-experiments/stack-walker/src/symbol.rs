//! Symbol resolution abstraction
//!
//! This module provides traits for resolving addresses to symbol information.
//! The actual symbol resolution is pluggable - you can implement the traits
//! to use addr2line, your own JIT symbol tables, or any other source.

use std::path::Path;

/// Information about a resolved symbol
#[derive(Debug, Clone, Default)]
pub struct SymbolInfo {
    /// Symbol/function name (mangled)
    pub name: Option<String>,

    /// Demangled name (if different from name)
    pub demangled_name: Option<String>,

    /// Source file path
    pub file: Option<String>,

    /// Line number in source
    pub line: Option<u32>,

    /// Column number in source
    pub column: Option<u32>,

    /// Start address of the function
    pub function_start: Option<u64>,

    /// Size of the function (if known)
    pub function_size: Option<u64>,

    /// Module/library name containing this symbol
    pub module: Option<String>,
}

impl SymbolInfo {
    /// Create an empty/unknown symbol info
    pub fn unknown() -> Self {
        Self::default()
    }

    /// Create with just a name
    pub fn with_name(name: impl Into<String>) -> Self {
        Self {
            name: Some(name.into()),
            ..Default::default()
        }
    }

    /// Get the best available display name
    pub fn display_name(&self) -> Option<&str> {
        self.demangled_name.as_deref().or(self.name.as_deref())
    }

    /// Check if any symbol information is available
    pub fn is_resolved(&self) -> bool {
        self.name.is_some() || self.file.is_some()
    }

    /// Get a formatted location string (file:line:column)
    pub fn location(&self) -> Option<String> {
        let file = self.file.as_ref()?;
        match (self.line, self.column) {
            (Some(line), Some(col)) => Some(format!("{}:{}:{}", file, line, col)),
            (Some(line), None) => Some(format!("{}:{}", file, line)),
            _ => Some(file.clone()),
        }
    }

    /// Calculate offset within function
    pub fn offset_in_function(&self, address: u64) -> Option<u64> {
        self.function_start.map(|start| address.saturating_sub(start))
    }
}

/// Trait for resolving addresses to symbols
///
/// Implement this trait to provide symbol resolution from your own sources.
///
/// # Example
///
/// ```ignore
/// struct JitSymbolResolver {
///     symbols: HashMap<u64, String>,
/// }
///
/// impl SymbolResolver for JitSymbolResolver {
///     fn resolve(&self, address: u64) -> Option<SymbolInfo> {
///         // Look up in JIT symbol table
///         self.symbols.get(&address).map(|name| {
///             SymbolInfo::with_name(name.clone())
///         })
///     }
/// }
/// ```
pub trait SymbolResolver {
    /// Resolve an address to symbol information
    ///
    /// Returns None if the address cannot be resolved.
    fn resolve(&self, address: u64) -> Option<SymbolInfo>;

    /// Resolve multiple addresses (batch optimization)
    ///
    /// Default implementation calls `resolve` for each address.
    /// Override for more efficient batch lookups.
    fn resolve_batch(&self, addresses: &[u64]) -> Vec<Option<SymbolInfo>> {
        addresses.iter().map(|&addr| self.resolve(addr)).collect()
    }

    /// Add a module/library for symbol resolution
    ///
    /// This is optional - some resolvers may not support dynamic module loading.
    fn add_module(&mut self, _path: &Path, _base_address: u64) -> Result<(), SymbolError> {
        Err(SymbolError::NotSupported("add_module not supported".into()))
    }

    /// Remove a module from resolution
    fn remove_module(&mut self, _base_address: u64) {
        // Default: no-op
    }
}

/// Errors during symbol resolution
#[derive(Debug, Clone)]
pub enum SymbolError {
    /// Module/file not found
    ModuleNotFound(String),
    /// Parse error in debug info
    ParseError(String),
    /// I/O error
    IoError(String),
    /// Operation not supported
    NotSupported(String),
}

impl std::fmt::Display for SymbolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SymbolError::ModuleNotFound(s) => write!(f, "Module not found: {}", s),
            SymbolError::ParseError(s) => write!(f, "Parse error: {}", s),
            SymbolError::IoError(s) => write!(f, "I/O error: {}", s),
            SymbolError::NotSupported(s) => write!(f, "Not supported: {}", s),
        }
    }
}

impl std::error::Error for SymbolError {}

/// A symbol resolver that always returns unknown
#[derive(Debug, Clone, Copy, Default)]
pub struct NullSymbolResolver;

impl SymbolResolver for NullSymbolResolver {
    fn resolve(&self, _address: u64) -> Option<SymbolInfo> {
        None
    }
}

/// A simple map-based symbol resolver
///
/// Useful for testing or simple JIT symbol tables.
#[derive(Debug, Clone, Default)]
pub struct MapSymbolResolver {
    symbols: std::collections::HashMap<u64, SymbolInfo>,
}

impl MapSymbolResolver {
    /// Create a new empty resolver
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a symbol
    pub fn add(&mut self, address: u64, info: SymbolInfo) {
        self.symbols.insert(address, info);
    }

    /// Add a simple named symbol
    pub fn add_named(&mut self, address: u64, name: impl Into<String>) {
        self.symbols.insert(address, SymbolInfo::with_name(name));
    }

    /// Number of symbols
    pub fn len(&self) -> usize {
        self.symbols.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.symbols.is_empty()
    }
}

impl SymbolResolver for MapSymbolResolver {
    fn resolve(&self, address: u64) -> Option<SymbolInfo> {
        self.symbols.get(&address).cloned()
    }
}

/// A resolver that chains multiple resolvers
///
/// Tries each resolver in order until one returns a result.
pub struct ChainedResolver<'a> {
    resolvers: Vec<&'a dyn SymbolResolver>,
}

impl<'a> ChainedResolver<'a> {
    /// Create a new chained resolver
    pub fn new(resolvers: Vec<&'a dyn SymbolResolver>) -> Self {
        Self { resolvers }
    }

    /// Add a resolver to the chain
    pub fn push(&mut self, resolver: &'a dyn SymbolResolver) {
        self.resolvers.push(resolver);
    }
}

impl<'a> SymbolResolver for ChainedResolver<'a> {
    fn resolve(&self, address: u64) -> Option<SymbolInfo> {
        for resolver in &self.resolvers {
            if let Some(info) = resolver.resolve(address) {
                return Some(info);
            }
        }
        None
    }
}

/// A resolver that looks up by range (function start to end)
#[derive(Debug, Clone, Default)]
pub struct RangeSymbolResolver {
    /// Sorted list of (start, end, info)
    ranges: Vec<(u64, u64, SymbolInfo)>,
}

impl RangeSymbolResolver {
    /// Create a new range resolver
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a symbol with its address range
    pub fn add_range(&mut self, start: u64, end: u64, mut info: SymbolInfo) {
        info.function_start = Some(start);
        info.function_size = Some(end - start);
        self.ranges.push((start, end, info));
        self.ranges.sort_by_key(|(start, _, _)| *start);
    }

    /// Add a symbol with just start and size
    pub fn add(&mut self, start: u64, size: u64, info: SymbolInfo) {
        self.add_range(start, start + size, info);
    }
}

impl SymbolResolver for RangeSymbolResolver {
    fn resolve(&self, address: u64) -> Option<SymbolInfo> {
        // Binary search for the range containing this address
        let idx = self.ranges.partition_point(|(start, _, _)| *start <= address);
        if idx == 0 {
            return None;
        }

        let (start, end, info) = &self.ranges[idx - 1];
        if address >= *start && address < *end {
            Some(info.clone())
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_symbol_info() {
        let info = SymbolInfo {
            name: Some("foo".into()),
            demangled_name: Some("Foo::bar".into()),
            file: Some("/src/foo.rs".into()),
            line: Some(42),
            column: Some(10),
            function_start: Some(0x1000),
            function_size: Some(0x100),
            module: None,
        };

        assert_eq!(info.display_name(), Some("Foo::bar"));
        assert_eq!(info.location(), Some("/src/foo.rs:42:10".into()));
        assert_eq!(info.offset_in_function(0x1050), Some(0x50));
        assert!(info.is_resolved());
    }

    #[test]
    fn test_map_resolver() {
        let mut resolver = MapSymbolResolver::new();
        resolver.add_named(0x1000, "main");
        resolver.add_named(0x2000, "foo");

        assert_eq!(resolver.resolve(0x1000).unwrap().display_name(), Some("main"));
        assert_eq!(resolver.resolve(0x2000).unwrap().display_name(), Some("foo"));
        assert!(resolver.resolve(0x3000).is_none());
    }

    #[test]
    fn test_range_resolver() {
        let mut resolver = RangeSymbolResolver::new();
        resolver.add(0x1000, 0x100, SymbolInfo::with_name("main"));
        resolver.add(0x2000, 0x200, SymbolInfo::with_name("foo"));

        // Within main
        assert_eq!(resolver.resolve(0x1050).unwrap().display_name(), Some("main"));
        // Within foo
        assert_eq!(resolver.resolve(0x2100).unwrap().display_name(), Some("foo"));
        // Before main
        assert!(resolver.resolve(0x0500).is_none());
        // Between functions
        assert!(resolver.resolve(0x1500).is_none());
        // After foo
        assert!(resolver.resolve(0x3000).is_none());
    }

    #[test]
    fn test_chained_resolver() {
        let mut r1 = MapSymbolResolver::new();
        r1.add_named(0x1000, "from_r1");

        let mut r2 = MapSymbolResolver::new();
        r2.add_named(0x2000, "from_r2");

        let chained = ChainedResolver::new(vec![&r1, &r2]);

        assert_eq!(chained.resolve(0x1000).unwrap().display_name(), Some("from_r1"));
        assert_eq!(chained.resolve(0x2000).unwrap().display_name(), Some("from_r2"));
        assert!(chained.resolve(0x3000).is_none());
    }
}
