/// Self-Contained Lisp Compilation System
///
/// This module implements a fully self-contained compilation pipeline
/// where everything is defined and controlled via Lisp code.
///
/// No Rust API needed - just write Lisp and it works!

use crate::{
    parser::{self, Value},
    macro_expander::MacroExpander,
    dialect_registry::DialectRegistry,
    mlir_context::MlirContext,
};
use melior::Context;

/// Self-contained compiler that processes Lisp source end-to-end
pub struct SelfContainedCompiler<'c> {
    context: &'c Context,
    mlir_ctx: MlirContext,
    expander: MacroExpander,
    registry: DialectRegistry,
    loaded_files: std::collections::HashSet<String>,
    search_paths: Vec<String>,
}

impl<'c> SelfContainedCompiler<'c> {
    pub fn new(context: &'c Context) -> Self {
        Self {
            context,
            mlir_ctx: MlirContext::new(),
            expander: MacroExpander::new(),
            registry: DialectRegistry::new(),
            loaded_files: std::collections::HashSet::new(),
            search_paths: vec![
                ".".to_string(),
                "./dialects".to_string(),
                "./lib".to_string(),
            ],
        }
    }

    /// Add a directory to the search path for imports
    pub fn add_search_path(&mut self, path: String) {
        self.search_paths.push(path);
    }

    /// Evaluate a Lisp expression in the compiler context
    /// This handles special forms like defirdl-dialect, deftransform, etc.
    pub fn eval(&mut self, expr: &Value) -> Result<Value, String> {
        // Check if this is a definition form that needs to be registered
        if let Value::List(elements) = expr {
            if let Some(Value::Symbol(s)) = elements.first() {
                match s.as_str() {
                    "defirdl-dialect" | "deftransform" | "defpdl-pattern" => {
                        // Expand the macro
                        let expanded = self.expander.expand(expr)?;

                        // Register with the dialect registry
                        self.registry.process_expanded_form(&expanded)?;

                        // Return the name of what was defined
                        if elements.len() >= 2 {
                            return Ok(elements[1].clone());
                        }
                        return Ok(Value::Symbol("ok".to_string()));
                    }
                    "import-dialect" | "import" => {
                        // Import a file or dialect
                        if elements.len() >= 2 {
                            let import_name = match &elements[1] {
                                Value::Symbol(s) | Value::String(s) => s.clone(),
                                _ => return Err("import requires a name".into()),
                            };

                            // Try to load the file
                            let result = self.load_import(&import_name)?;
                            return Ok(Value::List(vec![
                                Value::Symbol("imported".to_string()),
                                Value::String(import_name),
                                result,
                            ]));
                        }
                        return Err("import requires a name".into());
                    }
                    "list-dialects" => {
                        let dialects: Vec<Value> = self.registry.list_dialects()
                            .into_iter()
                            .map(|s| Value::String(s.to_string()))
                            .collect();
                        return Ok(Value::Vector(dialects));
                    }
                    "list-transforms" => {
                        let transforms: Vec<Value> = self.registry.list_transforms()
                            .into_iter()
                            .map(|s| Value::String(s.to_string()))
                            .collect();
                        return Ok(Value::Vector(transforms));
                    }
                    "list-patterns" => {
                        let patterns: Vec<Value> = self.registry.list_patterns()
                            .into_iter()
                            .map(|s| Value::String(s.to_string()))
                            .collect();
                        return Ok(Value::Vector(patterns));
                    }
                    "get-dialect" => {
                        if elements.len() >= 2 {
                            if let Value::String(name) = &elements[1] {
                                if let Some(dialect) = self.registry.get_dialect(name) {
                                    // Return dialect info as a map
                                    return Ok(Value::Map(vec![
                                        (Value::Keyword("name".to_string()), Value::String(dialect.name.clone())),
                                        (Value::Keyword("namespace".to_string()), Value::String(dialect.namespace.clone())),
                                        (Value::Keyword("description".to_string()), Value::String(dialect.description.clone())),
                                        (Value::Keyword("operations".to_string()), Value::Vector(
                                            dialect.operations.iter()
                                                .map(|op| Value::String(op.name.clone()))
                                                .collect()
                                        )),
                                    ]));
                                }
                            }
                        }
                        return Err("get-dialect requires a dialect name".into());
                    }
                    _ => {}
                }
            }
        }

        // For other expressions, just expand macros
        self.expander.expand(expr)
    }

    /// Load and evaluate a Lisp file
    pub fn load_file(&mut self, path: &str) -> Result<Value, String> {
        // Check if already loaded
        let canonical_path = std::fs::canonicalize(path)
            .map_err(|e| format!("Failed to resolve path {}: {}", path, e))?;

        let path_str = canonical_path.to_string_lossy().to_string();

        if self.loaded_files.contains(&path_str) {
            return Ok(Value::Symbol("already-loaded".to_string()));
        }

        let source = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read file {}: {}", path, e))?;

        self.loaded_files.insert(path_str);
        self.eval_string(&source)
    }

    /// Load an import by searching for it in search paths
    fn load_import(&mut self, name: &str) -> Result<Value, String> {
        // Try different file extensions and search paths
        let extensions = vec!["", ".lisp", ".mlir-lisp"];

        for search_path in &self.search_paths.clone() {
            for ext in &extensions {
                let file_path = format!("{}/{}{}", search_path, name, ext);

                if std::path::Path::new(&file_path).exists() {
                    return self.load_file(&file_path);
                }
            }
        }

        Err(format!("Could not find import '{}' in search paths {:?}", name, self.search_paths))
    }

    /// Evaluate a string of Lisp code
    pub fn eval_string(&mut self, source: &str) -> Result<Value, String> {
        let (_, values) = parser::parse(source)
            .map_err(|e| format!("Parse error: {:?}", e))?;

        let mut result = Value::Symbol("nil".to_string());

        for value in values {
            result = self.eval(&value)?;
        }

        Ok(result)
    }

    /// Get the dialect registry (for inspection)
    pub fn registry(&self) -> &DialectRegistry {
        &self.registry
    }

    /// Get the MLIR context
    pub fn mlir_context(&self) -> &MlirContext {
        &self.mlir_ctx
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_self_contained_dialect_definition() {
        let context = Context::new();
        let mut compiler = SelfContainedCompiler::new(&context);

        let source = r#"
(defirdl-dialect test
  :namespace "test"
  :description "Test dialect"

  (defirdl-op foo
    :summary "Foo operation"
    :results [(result AnyInteger)]))
"#;

        let result = compiler.eval_string(source);
        assert!(result.is_ok());

        // Check that the dialect was registered
        assert!(compiler.registry().get_dialect("test").is_some());
    }
}
