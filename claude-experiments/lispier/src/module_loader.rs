use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use thiserror::Error;

use crate::ast::{Node, Require};
use crate::jit::Jit;
use crate::macro_compiler::MacroCompiler;
use crate::macros::{MacroExpander, MacroRegistry};
use crate::parser::Parser;
use crate::reader::Reader;
use crate::tokenizer::Tokenizer;
use crate::value::Value;

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

    #[error("macro error in {0}: {1}")]
    MacroError(PathBuf, String),

    #[error("parser error in {0}: {1}")]
    ParserError(PathBuf, String),

    #[error("macro compilation error in {0}: {1}")]
    MacroCompilationError(PathBuf, String),

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

    /// Cache of already-compiled macro modules (canonical path -> JIT)
    /// We keep the JIT instances alive to prevent the compiled code from being dropped
    macro_jits: HashMap<PathBuf, Jit>,

    /// Macro compiler instance
    macro_compiler: MacroCompiler,
}

impl ModuleLoader {
    pub fn new(project_root: impl Into<PathBuf>) -> Self {
        Self {
            project_root: project_root.into(),
            cache: HashMap::new(),
            loading: HashSet::new(),
            macro_jits: HashMap::new(),
            macro_compiler: MacroCompiler::new(),
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
        let nodes = self.parse_source(&source, path, current_dir)?;

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

        // Filter out require and require-macros nodes from the result (they're metadata, not code)
        let filtered_nodes: Vec<Node> = nodes
            .into_iter()
            .filter(|n| !matches!(n, Node::Require(_) | Node::RequireMacros(_)))
            .collect();

        // Cache this module's nodes
        self.cache.insert(path.clone(), filtered_nodes.clone());

        self.loading.remove(path);

        all_nodes.extend(filtered_nodes);
        Ok(all_nodes)
    }

    /// Extract require-macros paths from a list of Values
    fn extract_require_macros_paths(&self, values: &[Value]) -> Vec<String> {
        let mut paths = Vec::new();

        for value in values {
            if let Value::List(items) = value {
                if items.len() >= 2 {
                    if let Some(Value::Symbol(sym)) = items.first() {
                        if sym.name == "require-macros" {
                            if let Some(Value::String(path)) = items.get(1) {
                                paths.push(path.clone());
                            }
                        }
                    }
                }
            }
        }

        paths
    }

    /// Load and compile macro modules, returning a MacroRegistry with all macros
    fn load_macro_modules(
        &mut self,
        macro_paths: &[String],
        current_dir: &Path,
        source_path: &PathBuf,
    ) -> Result<MacroRegistry, ModuleError> {
        let mut registry = MacroRegistry::new(); // Start with built-in macros

        for macro_path in macro_paths {
            let resolved = self.resolve_path(macro_path, current_dir)?;

            // Check if already compiled
            if self.macro_jits.contains_key(&resolved) {
                // Macros are already registered from a previous load
                continue;
            }

            // Read the macro module source
            let macro_source = fs::read_to_string(&resolved)
                .map_err(|_| ModuleError::FileNotFound(resolved.clone()))?;

            // Compile and register macros
            let jit = self
                .macro_compiler
                .compile_and_register(&macro_source, &mut registry)
                .map_err(|e| {
                    ModuleError::MacroCompilationError(
                        source_path.clone(),
                        format!("failed to compile macro module '{}': {}", macro_path, e),
                    )
                })?;

            // Keep the JIT alive
            self.macro_jits.insert(resolved, jit);
        }

        Ok(registry)
    }

    fn parse_source(
        &mut self,
        source: &str,
        path: &PathBuf,
        current_dir: &Path,
    ) -> Result<Vec<Node>, ModuleError> {
        let mut tokenizer = Tokenizer::new(source);
        let tokens = tokenizer
            .tokenize()
            .map_err(|e| ModuleError::TokenizerError(path.clone(), e.to_string()))?;

        let mut reader = Reader::new(&tokens);
        let values = reader
            .read()
            .map_err(|e| ModuleError::ReaderError(path.clone(), e.to_string()))?;

        // Extract require-macros paths and compile macro modules
        let macro_paths = self.extract_require_macros_paths(&values);
        let registry = self.load_macro_modules(&macro_paths, current_dir, path)?;

        // Macro expansion with the custom registry
        let mut expander = MacroExpander::with_registry(registry);
        let expanded = expander
            .expand_all(&values)
            .map_err(|e| ModuleError::MacroError(path.clone(), e.to_string()))?;

        // Transfer JIT instances from expander to keep dynamically compiled macros alive
        let dynamic_jits = expander.take_jit_instances();
        for jit in dynamic_jits {
            // Use the path as a key with a counter to avoid collisions
            let key = path.with_extension(format!(
                "dynamic_macro_{}",
                self.macro_jits.len()
            ));
            self.macro_jits.insert(key, jit);
        }

        let mut parser = Parser::new();
        parser
            .parse(&expanded)
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use tempfile::TempDir;

    #[test]
    fn test_require_macros_loading() {
        // Create a temp directory
        let temp_dir = TempDir::new().unwrap();
        let temp_path = temp_dir.path();

        // Create a macro module
        let macro_source = r#"
(require-dialect func)
(extern :value-ffi)

(func.func {:sym_name "value_list_first"
            :function_type (-> [!llvm.ptr] [!llvm.ptr])
            :sym_visibility "private"})

(func.func {:sym_name "identity"
            :function_type (-> [!llvm.ptr] [!llvm.ptr])}
  (do
    (block [(: form !llvm.ptr)]
      (def result (func.call {:callee @value_list_first :result !llvm.ptr} form))
      (func.return result))))

(defmacro identity)
"#;
        let macro_file = temp_path.join("macros.lisp");
        let mut f = fs::File::create(&macro_file).unwrap();
        f.write_all(macro_source.as_bytes()).unwrap();

        // Create a main module that uses the macro
        let main_source = r#"
(require-dialect func)
(require-macros "./macros.lisp")

(defn main [] -> i64
  (func.return (identity (: 42 i64))))
"#;
        let main_file = temp_path.join("main.lisp");
        let mut f = fs::File::create(&main_file).unwrap();
        f.write_all(main_source.as_bytes()).unwrap();

        // Load the main module
        let mut loader = ModuleLoader::new(temp_path);
        let nodes = loader.load(&main_file).expect("Failed to load main module");

        // Verify the module loaded correctly
        // The identity macro should have been expanded
        assert!(!nodes.is_empty());

        // Check that we have a func.func for main
        let has_main = nodes.iter().any(|n| {
            if let Node::Operation(op) = n {
                if op.qualified_name() == "func.func" {
                    if let Some(crate::ast::AttributeValue::String(name)) =
                        op.attributes.get("sym_name")
                    {
                        return name == "main";
                    }
                }
            }
            false
        });
        assert!(has_main, "Expected main function in nodes");
    }

    #[test]
    fn test_module_without_require_macros() {
        // Create a temp directory
        let temp_dir = TempDir::new().unwrap();
        let temp_path = temp_dir.path();

        // Create a simple module without require-macros
        let source = r#"
(require-dialect func)

(defn main [] -> i64
  (func.return (: 42 i64)))
"#;
        let main_file = temp_path.join("simple.lisp");
        let mut f = fs::File::create(&main_file).unwrap();
        f.write_all(source.as_bytes()).unwrap();

        // Load the module
        let mut loader = ModuleLoader::new(temp_path);
        let nodes = loader.load(&main_file).expect("Failed to load module");

        // Verify the module loaded correctly
        assert!(!nodes.is_empty());
    }
}
