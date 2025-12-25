use std::collections::HashMap;

/// Find a character in a string, but only if it's outside of angle brackets
fn find_outside_angles(s: &str, c: char) -> Option<usize> {
    let mut depth: i32 = 0;
    for (i, ch) in s.char_indices() {
        if ch == '<' {
            depth += 1;
        } else if ch == '>' {
            depth = depth.saturating_sub(1);
        } else if ch == c && depth == 0 {
            return Some(i);
        }
    }
    None
}

/// Namespace information for symbols
#[derive(Debug, Clone, PartialEq)]
pub struct Namespace {
    pub name: String,
    pub alias: Option<String>,
}

impl Namespace {
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            alias: None,
        }
    }

    pub fn with_alias(name: impl Into<String>, alias: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            alias: Some(alias.into()),
        }
    }
}

/// Namespace scope for tracking imports
pub struct NamespaceScope {
    /// Required dialects: name -> namespace
    required: HashMap<String, Namespace>,
    /// Used dialects (for unqualified access)
    used: Vec<Namespace>,
    /// Default namespace for unresolved symbols
    user_namespace: Namespace,
    /// Required files: alias -> file path
    required_files: HashMap<String, String>,
}

impl NamespaceScope {
    pub fn new() -> Self {
        Self {
            required: HashMap::new(),
            used: Vec::new(),
            user_namespace: Namespace::new("user"),
            required_files: HashMap::new(),
        }
    }

    /// Add a required dialect (require-dialect)
    pub fn require_dialect(&mut self, name: &str, alias: Option<&str>) {
        let ns = match alias {
            Some(a) => Namespace::with_alias(name, a),
            None => Namespace::new(name),
        };
        self.required.insert(name.to_string(), ns);
    }

    /// Add a used dialect (use-dialect)
    pub fn use_dialect(&mut self, name: &str) {
        let ns = Namespace::new(name);
        self.used.push(ns);
    }

    /// Add a required file (require)
    pub fn require_file(&mut self, path: &str, alias: &str) {
        self.required_files.insert(alias.to_string(), path.to_string());
        // Also register as a namespace for symbol resolution (alias -> path as namespace name)
        let ns = Namespace::with_alias(path, alias);
        self.required.insert(alias.to_string(), ns);
    }

    /// Resolve a symbol to its namespace
    /// Handles: alias/name, namespace.name, and bare names
    /// Returns the user namespace for local/unresolved symbols
    /// Returns Err with the unknown namespace name if a qualified symbol references an unknown dialect
    pub fn resolve_symbol(&mut self, symbol_text: &str) -> Result<Namespace, String> {
        // Check for slash notation: alias/name (only before any angle bracket)
        if let Some(slash_pos) = find_outside_angles(symbol_text, '/') {
            let alias = &symbol_text[0..slash_pos];
            // Find namespace with this alias
            for ns in self.required.values() {
                if let Some(ref ns_alias) = ns.alias {
                    if ns_alias == alias {
                        return Ok(ns.clone());
                    }
                }
            }
            // Alias not found - create a namespace for it
            // This allows struct accessors like Point/x to work without requiring Point as a dialect
            return Ok(Namespace::with_alias(alias, alias));
        }

        // Check for dot notation: namespace.name (only before any angle bracket)
        if let Some(dot_pos) = find_outside_angles(symbol_text, '.') {
            let namespace_name = &symbol_text[0..dot_pos];

            // Check if already in required
            if let Some(ns) = self.required.get(namespace_name) {
                return Ok(ns.clone());
            }

            // Check if in used (from use-dialect)
            for ns in &self.used {
                if ns.name == namespace_name {
                    return Ok(ns.clone());
                }
            }

            // Not found anywhere - this is an error, dialect must be required
            return Err(namespace_name.to_string());
        }

        // Bare name - search used dialects
        // For now, without a dialect registry, we just return user namespace
        // In the full implementation, we'd check which dialect contains this operation
        Ok(self.user_namespace.clone())
    }

    /// Get the unqualified part of a symbol (after / or .)
    pub fn get_unqualified_name<'a>(&self, symbol_text: &'a str) -> &'a str {
        if let Some(slash_pos) = find_outside_angles(symbol_text, '/') {
            return &symbol_text[slash_pos + 1..];
        }
        if let Some(dot_pos) = find_outside_angles(symbol_text, '.') {
            return &symbol_text[dot_pos + 1..];
        }
        symbol_text
    }

    /// Get the user namespace
    pub fn user_namespace(&self) -> &Namespace {
        &self.user_namespace
    }
}

impl Default for NamespaceScope {
    fn default() -> Self {
        Self::new()
    }
}
