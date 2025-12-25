//! Macro expander - recursively expands macros in Value trees
//!
//! This module handles macro expansion with support for dynamic macro definition.
//! Macros can define other macros that become available for subsequent code.

use std::collections::HashMap;

use super::builtins::{StructFieldGetMacro, StructFieldSetMacro};
use super::{MacroError, MacroRegistry};
use crate::jit::Jit;
use crate::macro_compiler::MacroCompiler;
use crate::value::Value;

/// Maximum depth for macro expansion to prevent infinite loops
const MAX_EXPANSION_DEPTH: usize = 100;

/// Parsed struct definition from defstruct form
struct StructDef {
    name: String,
    fields: Vec<(String, String)>, // (field_name, field_type)
}

/// Context for dynamic macro compilation
/// Tracks accumulated definitions that may be needed to compile new macros
#[derive(Default)]
struct DynamicMacroContext {
    /// Accumulated require-dialect declarations
    require_dialects: Vec<Value>,
    /// Accumulated extern declarations
    externs: Vec<Value>,
    /// Accumulated func.func declarations (potential macro implementations)
    func_funcs: Vec<Value>,
    /// JIT instances to keep alive (prevents compiled code from being dropped)
    jit_instances: Vec<Jit>,
}

impl DynamicMacroContext {
    fn new() -> Self {
        Self::default()
    }

    /// Check if a value is a require-dialect declaration
    fn is_require_dialect(value: &Value) -> bool {
        if let Value::List(items) = value {
            if let Some(Value::Symbol(sym)) = items.first() {
                return sym.name == "require-dialect";
            }
        }
        false
    }

    /// Check if a value is an extern declaration
    fn is_extern(value: &Value) -> bool {
        if let Value::List(items) = value {
            if let Some(Value::Symbol(sym)) = items.first() {
                return sym.name == "extern";
            }
        }
        false
    }

    /// Check if a value is a func.func declaration
    fn is_func_func(value: &Value) -> bool {
        if let Value::List(items) = value {
            if let Some(Value::Symbol(sym)) = items.first() {
                return sym.name == "func.func" || sym.qualified_name() == "func.func";
            }
        }
        false
    }

    /// Check if a value is a defmacro declaration
    fn is_defmacro(value: &Value) -> Option<String> {
        if let Value::List(items) = value {
            if items.len() >= 2 {
                if let Some(Value::Symbol(sym)) = items.first() {
                    if sym.name == "defmacro" {
                        if let Some(Value::Symbol(name_sym)) = items.get(1) {
                            return Some(name_sym.name.clone());
                        }
                    }
                }
            }
        }
        None
    }

    /// Extract the sym_name from a func.func declaration
    fn get_func_name(value: &Value) -> Option<String> {
        if let Value::List(items) = value {
            // Look for the attributes map (second element)
            if let Some(Value::Map(attrs)) = items.get(1) {
                if let Some(Value::String(name)) = attrs.get("sym_name") {
                    return Some(name.clone());
                }
            }
        }
        None
    }

    /// Add a value to context if it's relevant
    fn accumulate(&mut self, value: &Value) {
        if Self::is_require_dialect(value) {
            // Only add if not already present
            if !self.require_dialects.iter().any(|v| v == value) {
                self.require_dialects.push(value.clone());
            }
        } else if Self::is_extern(value) {
            self.externs.push(value.clone());
        } else if Self::is_func_func(value) {
            self.func_funcs.push(value.clone());
        }
    }

    /// Build the source for compiling a specific macro
    /// Returns the values needed to compile the macro with the given name
    fn build_macro_source(&self, macro_name: &str) -> Vec<Value> {
        let mut source = Vec::new();

        // Add require-dialect declarations first
        source.extend(self.require_dialects.clone());

        // Add all extern declarations
        source.extend(self.externs.clone());

        // Add all func.func declarations (FFI declarations + macro implementation)
        // We need to include FFI declarations because the macro may call them
        let mut found_macro = false;
        for func in &self.func_funcs {
            source.push(func.clone());
            if let Some(name) = Self::get_func_name(func) {
                if name == macro_name {
                    found_macro = true;
                }
            }
        }

        // Only add defmacro if we found the macro function
        if found_macro {
            source.push(Value::List(vec![
                Value::symbol("defmacro"),
                Value::symbol(macro_name),
            ]));
        }

        source
    }

    /// Store a JIT instance to keep it alive
    fn keep_jit_alive(&mut self, jit: Jit) {
        self.jit_instances.push(jit);
    }
}

/// Macro expander that recursively expands macros in Value trees
///
/// Supports dynamic macro definition: macros can define other macros
/// that become available for subsequent expansion.
pub struct MacroExpander {
    registry: MacroRegistry,
    context: DynamicMacroContext,
    compiler: MacroCompiler,
    /// When true, defmacro forms trigger dynamic compilation.
    /// When false, defmacro forms pass through (for macro module compilation).
    enable_dynamic_macros: bool,
    /// Struct sizes for (new StructName) expansion
    struct_sizes: HashMap<String, usize>,
}

impl MacroExpander {
    /// Create a new expander with the default macro registry
    /// Dynamic macro compilation is enabled by default.
    pub fn new() -> Self {
        Self {
            registry: MacroRegistry::new(),
            context: DynamicMacroContext::new(),
            compiler: MacroCompiler::new(),
            enable_dynamic_macros: true,
            struct_sizes: HashMap::new(),
        }
    }

    /// Create an expander with a custom registry
    /// Dynamic macro compilation is enabled by default.
    pub fn with_registry(registry: MacroRegistry) -> Self {
        Self {
            registry,
            context: DynamicMacroContext::new(),
            compiler: MacroCompiler::new(),
            enable_dynamic_macros: true,
            struct_sizes: HashMap::new(),
        }
    }

    /// Create an expander without dynamic macro compilation
    ///
    /// Use this when compiling macro modules, where defmacro declarations
    /// should be passed through for later processing by the macro compiler.
    pub fn new_without_dynamic_macros() -> Self {
        Self {
            registry: MacroRegistry::new(),
            context: DynamicMacroContext::new(),
            compiler: MacroCompiler::new(),
            enable_dynamic_macros: false,
            struct_sizes: HashMap::new(),
        }
    }

    /// Get a reference to the macro registry
    pub fn registry(&self) -> &MacroRegistry {
        &self.registry
    }

    /// Get a mutable reference to the macro registry
    pub fn registry_mut(&mut self) -> &mut MacroRegistry {
        &mut self.registry
    }

    /// Take ownership of the JIT instances (to transfer to caller for lifetime management)
    pub fn take_jit_instances(&mut self) -> Vec<Jit> {
        std::mem::take(&mut self.context.jit_instances)
    }

    /// Expand all macros in a slice of values
    ///
    /// This processes values sequentially, allowing macros defined earlier
    /// to be used by code that comes later.
    pub fn expand_all(&mut self, values: &[Value]) -> Result<Vec<Value>, MacroError> {
        let mut result = Vec::new();

        for value in values {
            // First, expand the value
            let expanded = self.expand_value(value, 0)?;

            // Process the expanded value for dynamic macro definitions
            let processed = self.process_for_defmacro(expanded)?;

            // Add non-empty results
            if !Self::is_empty_value(&processed) {
                result.push(processed);
            }
        }

        Ok(result)
    }

    /// Check if a value is "empty" (should be filtered out)
    fn is_empty_value(value: &Value) -> bool {
        matches!(value, Value::List(items) if items.is_empty())
    }

    /// Process an expanded value for defmacro declarations
    ///
    /// If the value contains defmacro and dynamic macros are enabled,
    /// compile and register the macro.
    /// Also accumulates context (func.funcs, externs) for future compilation.
    fn process_for_defmacro(&mut self, value: Value) -> Result<Value, MacroError> {
        // If dynamic macros are disabled, just pass through
        if !self.enable_dynamic_macros {
            return Ok(value);
        }

        // Handle `do` blocks - process each element
        if let Value::List(ref items) = value {
            if let Some(Value::Symbol(sym)) = items.first() {
                if sym.name == "do" {
                    return self.process_do_block(items);
                }
            }
        }

        // Check if this is a defmacro declaration
        if let Some(macro_name) = DynamicMacroContext::is_defmacro(&value) {
            // Compile and register the macro
            self.compile_and_register_macro(&macro_name)?;
            // Return empty list (defmacro is consumed)
            return Ok(Value::List(vec![]));
        }

        // Check if this is a defstruct declaration
        if let Some(struct_def) = Self::parse_defstruct(&value) {
            // Register accessor macros for the struct
            self.register_struct_macros(&struct_def)?;
            // Return empty list (defstruct is consumed)
            return Ok(Value::List(vec![]));
        }

        // Accumulate context for future macro compilation
        self.context.accumulate(&value);

        Ok(value)
    }

    /// Parse a defstruct form: (defstruct Name [field1 type1] [field2 type2] ...)
    fn parse_defstruct(value: &Value) -> Option<StructDef> {
        let items = match value {
            Value::List(items) => items,
            _ => return None,
        };

        // Check first element is 'defstruct'
        match items.first() {
            Some(Value::Symbol(sym)) if sym.name == "defstruct" => {}
            _ => return None,
        }

        // Second element is the struct name
        let name = match items.get(1) {
            Some(Value::Symbol(sym)) => sym.name.clone(),
            _ => return None,
        };

        // Remaining elements are field definitions: [field_name type]
        let mut fields = Vec::new();
        for field_def in items.iter().skip(2) {
            match field_def {
                Value::Vector(field_items) if field_items.len() == 2 => {
                    let field_name = match &field_items[0] {
                        Value::Symbol(sym) => sym.name.clone(),
                        _ => return None,
                    };
                    let field_type = match &field_items[1] {
                        Value::Symbol(sym) => sym.name.clone(),
                        _ => return None,
                    };
                    fields.push((field_name, field_type));
                }
                _ => return None,
            }
        }

        if fields.is_empty() {
            return None;
        }

        Some(StructDef { name, fields })
    }

    /// Register accessor macros for a struct
    fn register_struct_macros(&mut self, struct_def: &StructDef) -> Result<(), MacroError> {
        // Build the LLVM struct type string
        let field_types: Vec<&str> = struct_def.fields.iter().map(|(_, t)| t.as_str()).collect();
        let struct_type = format!("!llvm.struct<({})>", field_types.join(", "));

        // Calculate struct size (simplified - assumes 8 bytes per field for now)
        let struct_size = struct_def.fields.len() * 8;

        // Register getter and setter macros for each field
        for (index, (field_name, field_type)) in struct_def.fields.iter().enumerate() {
            // Getter: Point/x
            let getter_name = format!("{}/{}", struct_def.name, field_name);
            let getter = StructFieldGetMacro::new(
                &getter_name,
                &struct_type,
                index,
                field_type,
            );
            self.registry.register(Box::new(getter));

            // Setter: Point/x!
            let setter_name = format!("{}/{}!", struct_def.name, field_name);
            let setter = StructFieldSetMacro::new(
                &setter_name,
                &struct_type,
                index,
            );
            self.registry.register(Box::new(setter));
        }

        // Store struct size for (new StructName) expansion
        self.struct_sizes.insert(struct_def.name.clone(), struct_size);

        Ok(())
    }

    /// Process a `do` block, handling defmacros within it
    fn process_do_block(&mut self, items: &[Value]) -> Result<Value, MacroError> {
        let mut result_items = vec![Value::symbol("do")];

        for item in items.iter().skip(1) {
            // Recursively process each item
            let processed = self.process_for_defmacro(item.clone())?;
            if !Self::is_empty_value(&processed) {
                result_items.push(processed);
            }
        }

        // If the do block only has the "do" symbol left, return empty
        if result_items.len() == 1 {
            return Ok(Value::List(vec![]));
        }

        // If only one item left besides "do", unwrap it
        if result_items.len() == 2 {
            return Ok(result_items.pop().unwrap());
        }

        Ok(Value::List(result_items))
    }

    /// Compile and register a macro from accumulated context
    fn compile_and_register_macro(&mut self, macro_name: &str) -> Result<(), MacroError> {
        let source_values = self.context.build_macro_source(macro_name);

        // Check if the source includes the defmacro (meaning we found the func.func)
        let has_defmacro = source_values.iter().any(|v| {
            DynamicMacroContext::is_defmacro(v).is_some()
        });

        if !has_defmacro {
            return Err(MacroError::DynamicMacroCompilationFailed {
                macro_name: macro_name.to_string(),
                message: format!("no func.func found with name '{}'", macro_name),
            });
        }

        // Compile the macro
        let compiled = self.compiler.compile_from_values(&source_values).map_err(|e| {
            MacroError::DynamicMacroCompilationFailed {
                macro_name: macro_name.to_string(),
                message: e.to_string(),
            }
        })?;

        let (macros, jit) = compiled.into_macros();

        // Register the compiled macros
        for macro_impl in macros {
            self.registry.register(Box::new(macro_impl));
        }

        // Keep JIT alive
        self.context.keep_jit_alive(jit);

        Ok(())
    }

    /// Expand macros in a single value
    fn expand_value(&self, value: &Value, depth: usize) -> Result<Value, MacroError> {
        if depth > MAX_EXPANSION_DEPTH {
            return Err(MacroError::MaxDepthExceeded);
        }

        match value {
            Value::List(items) => {
                if let Some(expanded) = self.try_macro_expand(items)? {
                    // Re-expand the result (may contain more macros)
                    self.expand_value(&expanded, depth + 1)
                } else {
                    // Not a macro call - recursively expand children
                    let expanded_items: Vec<Value> = items
                        .iter()
                        .map(|item| self.expand_value(item, depth))
                        .collect::<Result<_, _>>()?;
                    Ok(Value::List(expanded_items))
                }
            }
            Value::Vector(items) => {
                let expanded: Vec<Value> = items
                    .iter()
                    .map(|item| self.expand_value(item, depth))
                    .collect::<Result<_, _>>()?;
                Ok(Value::Vector(expanded))
            }
            // Atoms pass through unchanged
            _ => Ok(value.clone()),
        }
    }

    /// Try to expand a list as a macro call
    ///
    /// Returns Some(expanded) if the first element is a macro name,
    /// None if it's not a macro call.
    fn try_macro_expand(&self, items: &[Value]) -> Result<Option<Value>, MacroError> {
        // Check if first element is a symbol that names a macro
        if let Some(Value::Symbol(sym)) = items.first() {
            // Handle (new StructName) - struct constructor
            if sym.name == "new" {
                if let Some(Value::Symbol(struct_sym)) = items.get(1) {
                    if let Some(&size) = self.struct_sizes.get(&struct_sym.name) {
                        // Generate: (func.call {:callee @malloc :result !llvm.ptr} (: size i64))
                        let malloc_call = Value::List(vec![
                            Value::symbol("func.call"),
                            Value::Map(
                                [
                                    ("callee".to_string(), Value::symbol("@malloc")),
                                    ("result".to_string(), Value::symbol("!llvm.ptr")),
                                ]
                                .into_iter()
                                .collect(),
                            ),
                            Value::List(vec![
                                Value::symbol(":"),
                                Value::Number(size as f64),
                                Value::symbol("i64"),
                            ]),
                        ]);
                        return Ok(Some(malloc_call));
                    }
                }
            }

            // For struct accessor macros (Point/x), use the qualified name
            // For regular macros, check if it's not dialect-qualified
            let is_dialect_qualified = sym
                .namespace
                .as_ref()
                .map(|ns| ns.name != "user")
                .unwrap_or(false);

            // First try to look up by qualified name (for struct accessors like Point/x)
            if sym.uses_alias {
                let qualified = sym.qualified_name();
                if let Some(macro_impl) = self.registry.get(&qualified) {
                    let args = &items[1..];
                    let expanded = macro_impl.expand(args)?;
                    return Ok(Some(expanded));
                }
            }

            // Then try by simple name (for regular macros)
            if !is_dialect_qualified {
                if let Some(macro_impl) = self.registry.get(&sym.name) {
                    // Expand the macro with the remaining items as arguments
                    let args = &items[1..];
                    let expanded = macro_impl.expand(args)?;
                    return Ok(Some(expanded));
                }
            }
        }
        Ok(None)
    }

    /// Check if a symbol names a macro
    pub fn is_macro(&self, name: &str) -> bool {
        self.registry.contains(name)
    }
}

impl Default for MacroExpander {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_non_macro_passthrough() {
        let mut expander = MacroExpander::new();
        let input = vec![Value::List(vec![
            Value::symbol("arith.addi"),
            Value::Number(1.0),
            Value::Number(2.0),
        ])];

        let result = expander.expand_all(&input).unwrap();
        assert_eq!(result.len(), 1);

        // Should pass through unchanged
        if let Value::List(items) = &result[0] {
            assert_eq!(items.len(), 3);
            if let Value::Symbol(sym) = &items[0] {
                assert_eq!(sym.name, "arith.addi");
            } else {
                panic!("Expected symbol");
            }
        } else {
            panic!("Expected list");
        }
    }

    #[test]
    fn test_nested_expansion() {
        let mut expander = MacroExpander::new();

        // Nested non-macro lists should have their contents expanded
        let input = vec![Value::List(vec![
            Value::symbol("outer"),
            Value::List(vec![Value::symbol("inner"), Value::Number(1.0)]),
        ])];

        let result = expander.expand_all(&input).unwrap();
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_defmacro_detection() {
        // Test that defmacro is properly detected
        let defmacro_value = Value::List(vec![
            Value::symbol("defmacro"),
            Value::symbol("my-macro"),
        ]);

        assert_eq!(
            DynamicMacroContext::is_defmacro(&defmacro_value),
            Some("my-macro".to_string())
        );

        // Non-defmacro should return None
        let other_value = Value::List(vec![Value::symbol("func.func")]);
        assert_eq!(DynamicMacroContext::is_defmacro(&other_value), None);
    }

    #[test]
    fn test_func_func_detection() {
        // Test func.func detection
        let func_value = Value::List(vec![
            Value::symbol("func.func"),
            Value::Map(std::collections::HashMap::from([(
                "sym_name".to_string(),
                Value::String("my-func".to_string()),
            )])),
        ]);

        assert!(DynamicMacroContext::is_func_func(&func_value));
        assert_eq!(
            DynamicMacroContext::get_func_name(&func_value),
            Some("my-func".to_string())
        );
    }
}
