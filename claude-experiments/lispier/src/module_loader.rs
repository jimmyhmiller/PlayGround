use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;

use crate::ast::{Node, Require};
use crate::parser::Parser;
use crate::reader::Reader;
use crate::tokenizer::Tokenizer;

#[derive(Debug, Error)]
pub enum ModuleError {
    #[error("file not found: {0}")]
    FileNotFound(PathBuf),

    #[error("circular dependency detected: {0}")]
    CircularDependency(String),

    #[error("tokenizer error in {0}: {1}")]
    TokenizerError(PathBuf, String),

    #[error("reader error in {0}: {1}")]
    ReaderError(PathBuf, String),

    #[error("parser error in {0}: {1}")]
    ParserError(PathBuf, String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),
}

/// Module loader handles file resolution, caching, and dependency ordering
pub struct ModuleLoader {
    /// Project root directory (for @ paths)
    project_root: PathBuf,

    /// Cache of already-loaded modules (canonical path -> parsed nodes)
    cache: HashMap<PathBuf, Vec<Node>>,

    /// Currently loading modules (for cycle detection)
    loading: HashSet<PathBuf>,
}

impl ModuleLoader {
    pub fn new(project_root: impl Into<PathBuf>) -> Self {
        Self {
            project_root: project_root.into(),
            cache: HashMap::new(),
            loading: HashSet::new(),
        }
    }

    /// Load a module and all its dependencies, returning merged AST nodes
    pub fn load(&mut self, entry_path: &Path) -> Result<Vec<Node>, ModuleError> {
        let canonical = entry_path
            .canonicalize()
            .map_err(|_| ModuleError::FileNotFound(entry_path.to_path_buf()))?;
        let current_dir = canonical
            .parent()
            .unwrap_or(Path::new("."))
            .to_path_buf();
        self.load_recursive(&canonical, &current_dir)
    }

    /// Resolve a require path to a canonical path
    fn resolve_path(&self, require_path: &str, current_dir: &Path) -> Result<PathBuf, ModuleError> {
        let resolved = if require_path.starts_with('@') {
            // Project-relative path: @lib/foo.lisp -> project_root/lib/foo.lisp
            self.project_root.join(&require_path[1..])
        } else if require_path.starts_with("./") || require_path.starts_with("../") {
            // Relative to current file
            current_dir.join(require_path)
        } else {
            // Treat as relative to current file
            current_dir.join(require_path)
        };

        resolved
            .canonicalize()
            .map_err(|_| ModuleError::FileNotFound(resolved))
    }

    /// Recursively load a module and its dependencies
    fn load_recursive(
        &mut self,
        path: &PathBuf,
        current_dir: &Path,
    ) -> Result<Vec<Node>, ModuleError> {
        // Check cache first - if already loaded, return empty (nodes are already in the result)
        if self.cache.contains_key(path) {
            return Ok(vec![]);
        }

        // Check for circular dependency
        if self.loading.contains(path) {
            return Err(ModuleError::CircularDependency(path.display().to_string()));
        }

        self.loading.insert(path.clone());

        // Read and parse the file
        let source = fs::read_to_string(path)?;
        let nodes = self.parse_source(&source, path)?;

        // Collect all require nodes and their resolved paths
        let mut all_nodes = Vec::new();
        let mut requires: Vec<(Require, PathBuf)> = Vec::new();

        for node in &nodes {
            if let Node::Require(req) = node {
                let resolved = self.resolve_path(&req.path, current_dir)?;
                requires.push((req.clone(), resolved));
            }
        }

        // Load dependencies first (depth-first)
        for (_, dep_path) in &requires {
            let dep_dir = dep_path.parent().unwrap().to_path_buf();
            let dep_nodes = self.load_recursive(dep_path, &dep_dir)?;
            all_nodes.extend(dep_nodes);
        }

        // Filter out require nodes from the result (they're metadata, not code)
        let filtered_nodes: Vec<Node> = nodes
            .into_iter()
            .filter(|n| !matches!(n, Node::Require(_)))
            .collect();

        // Cache this module's nodes
        self.cache.insert(path.clone(), filtered_nodes.clone());

        self.loading.remove(path);

        all_nodes.extend(filtered_nodes);
        Ok(all_nodes)
    }

    fn parse_source(&self, source: &str, path: &PathBuf) -> Result<Vec<Node>, ModuleError> {
        let mut tokenizer = Tokenizer::new(source);
        let tokens = tokenizer
            .tokenize()
            .map_err(|e| ModuleError::TokenizerError(path.clone(), e.to_string()))?;

        let mut reader = Reader::new(&tokens);
        let values = reader
            .read()
            .map_err(|e| ModuleError::ReaderError(path.clone(), e.to_string()))?;

        let mut parser = Parser::new();
        parser
            .parse(&values)
            .map_err(|e| ModuleError::ParserError(path.clone(), e.to_string()))
    }
}

/// Find project root by looking for a marker file (e.g., Cargo.toml, .git, or lispier.toml)
pub fn find_project_root(start: &Path) -> Option<PathBuf> {
    // Canonicalize the starting path to handle relative paths
    let canonical = start.canonicalize().ok()?;
    let mut current = canonical.as_path();

    // First try parent if start is a file
    if current.is_file() {
        current = current.parent()?;
    }

    loop {
        // Check for project markers
        if current.join("Cargo.toml").exists()
            || current.join("lispier.toml").exists()
            || current.join(".git").exists()
        {
            return Some(current.to_path_buf());
        }

        // Move to parent directory
        match current.parent() {
            Some(parent) if !parent.as_os_str().is_empty() => current = parent,
            _ => return None,
        }
    }
}
