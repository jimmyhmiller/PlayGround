//! Macro compiler for JIT-compiling macro modules
//!
//! This module handles the compilation of macro modules to create JitMacro instances
//! that can be registered with the macro expander.

use std::path::Path;
use thiserror::Error;

use crate::ast::Node;
use crate::dialect::DialectRegistry;
use crate::ir_gen::IRGenerator;
use crate::jit::Jit;
use crate::macros::{JitMacro, MacroExpander, MacroRegistry};
use crate::parser::Parser;
use crate::reader::Reader;
use crate::runtime::{extract_defmacros, extract_externs};
use crate::tokenizer::Tokenizer;

#[derive(Debug, Error)]
pub enum MacroCompilerError {
    #[error("failed to read file: {0}")]
    IoError(#[from] std::io::Error),

    #[error("tokenizer error: {0}")]
    TokenizerError(String),

    #[error("reader error: {0}")]
    ReaderError(String),

    #[error("macro expansion error: {0}")]
    MacroError(String),

    #[error("parser error: {0}")]
    ParserError(String),

    #[error("IR generation error: {0}")]
    IrGenError(String),

    #[error("JIT compilation error: {0}")]
    JitError(String),

    #[error("macro function not found: {0}")]
    MacroFunctionNotFound(String),

    #[error("unknown extern library: {0}")]
    UnknownExternLibrary(String),
}

/// Result of compiling a macro module
pub struct CompiledMacros {
    /// The JIT-compiled macros
    pub macros: Vec<JitMacro>,
    /// The JIT instance (kept alive to prevent dropping the compiled code)
    #[allow(dead_code)]
    jit: Jit,
}

impl CompiledMacros {
    /// Get the compiled macros
    pub fn into_macros(self) -> (Vec<JitMacro>, Jit) {
        (self.macros, self.jit)
    }
}

/// Compiles macro modules to JitMacro instances
pub struct MacroCompiler {
    registry: DialectRegistry,
}

impl MacroCompiler {
    /// Create a new macro compiler
    pub fn new() -> Self {
        Self {
            registry: DialectRegistry::new(),
        }
    }

    /// Compile a macro from Values (func.funcs, externs, and defmacro)
    ///
    /// This is used for dynamic macro compilation during expansion.
    /// Takes accumulated context (extern declarations and func.funcs) and compiles them.
    pub fn compile_from_values(
        &self,
        values: &[crate::value::Value],
    ) -> Result<CompiledMacros, MacroCompilerError> {
        // Serialize values back to source
        let source = values
            .iter()
            .map(|v| v.to_string())
            .collect::<Vec<_>>()
            .join("\n");

        // Compile using the normal source compilation path
        self.compile_source(&source)
    }

    /// Compile a macro module from source code
    ///
    /// This parses the source (using only built-in macros), extracts defmacro
    /// declarations, compiles to JIT, and returns the resulting JitMacros.
    pub fn compile_source(&self, source: &str) -> Result<CompiledMacros, MacroCompilerError> {
        // Tokenize
        let mut tokenizer = Tokenizer::new(source);
        let tokens = tokenizer
            .tokenize()
            .map_err(|e| MacroCompilerError::TokenizerError(e.to_string()))?;

        // Read
        let mut reader = Reader::new(&tokens);
        let values = reader
            .read()
            .map_err(|e| MacroCompilerError::ReaderError(e.to_string()))?;

        // Macro expansion (using only built-in macros, no dynamic macro compilation)
        // Dynamic macros are disabled because macro modules handle defmacro declarations
        // after parsing, not during expansion.
        let mut expander = MacroExpander::new_without_dynamic_macros();
        let expanded = expander
            .expand_all(&values)
            .map_err(|e| MacroCompilerError::MacroError(e.to_string()))?;

        // Parse
        let mut parser = Parser::new();
        let nodes = parser
            .parse(&expanded)
            .map_err(|e| MacroCompilerError::ParserError(e.to_string()))?;

        self.compile_nodes(&nodes)
    }

    /// Compile a macro module from a file path
    pub fn compile_file(&self, path: &Path) -> Result<CompiledMacros, MacroCompilerError> {
        let source = std::fs::read_to_string(path)?;
        self.compile_source(&source)
    }

    /// Compile a macro module from already-parsed AST nodes
    pub fn compile_nodes(&self, nodes: &[Node]) -> Result<CompiledMacros, MacroCompilerError> {
        // Extract defmacro declarations
        let defmacros = extract_defmacros(nodes);

        if defmacros.is_empty() {
            return Ok(CompiledMacros {
                macros: Vec::new(),
                jit: self.create_empty_jit()?,
            });
        }

        // Extract extern declarations
        let externs = extract_externs(nodes);

        // Generate MLIR
        let generator = IRGenerator::new(&self.registry);
        let mut module = generator
            .generate(nodes)
            .map_err(|e| MacroCompilerError::IrGenError(e.to_string()))?;

        // Create JIT
        let jit = Jit::new(&self.registry, &mut module)
            .map_err(|e| MacroCompilerError::JitError(e.to_string()))?;

        // Register FFI symbols based on extern declarations
        for ext in &externs {
            let lib = ext.library.trim_start_matches(':');
            match lib {
                "value-ffi" => unsafe {
                    jit.register_value_ffi();
                },
                other => {
                    return Err(MacroCompilerError::UnknownExternLibrary(other.to_string()));
                }
            }
        }

        // Look up macro functions and create JitMacros
        let mut macros = Vec::new();
        for defmacro in &defmacros {
            let macro_fn = jit
                .lookup_macro_fn(&defmacro.name)
                .ok_or_else(|| MacroCompilerError::MacroFunctionNotFound(defmacro.name.clone()))?;

            let jit_macro = unsafe { JitMacro::new(&defmacro.name, macro_fn) };
            macros.push(jit_macro);
        }

        Ok(CompiledMacros { macros, jit })
    }

    /// Compile a macro module and register the macros with a registry
    pub fn compile_and_register(
        &self,
        source: &str,
        registry: &mut MacroRegistry,
    ) -> Result<Jit, MacroCompilerError> {
        let compiled = self.compile_source(source)?;
        let (macros, jit) = compiled.into_macros();

        for macro_impl in macros {
            registry.register(Box::new(macro_impl));
        }

        Ok(jit)
    }

    /// Create an empty JIT (for modules with no macros)
    fn create_empty_jit(&self) -> Result<Jit, MacroCompilerError> {
        let mut module = self.registry.create_module();
        Jit::new(&self.registry, &mut module)
            .map_err(|e| MacroCompilerError::JitError(e.to_string()))
    }
}

impl Default for MacroCompiler {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::macros::Macro;
    use crate::value::Value;

    #[test]
    fn test_compile_identity_macro() {
        let source = r#"
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

        let compiler = MacroCompiler::new();
        let compiled = compiler.compile_source(source).expect("Compilation failed");

        assert_eq!(compiled.macros.len(), 1);
        assert_eq!(compiled.macros[0].name(), "identity");

        // Test the macro
        let args = vec![Value::Number(42.0)];
        let result = compiled.macros[0].expand(&args).expect("Expansion failed");
        assert!(matches!(result, Value::Number(n) if n == 42.0));
    }

    #[test]
    fn test_compile_empty_module() {
        let source = r#"
(require-dialect func)
(func.func {:sym_name "foo"
            :function_type (-> [] [i64])}
  (do
    (block
      (func.return (: 42 i64)))))
"#;

        let compiler = MacroCompiler::new();
        let compiled = compiler.compile_source(source).expect("Compilation failed");

        assert!(compiled.macros.is_empty());
    }
}
