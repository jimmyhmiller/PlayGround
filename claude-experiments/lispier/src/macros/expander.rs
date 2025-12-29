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
}

/// Collector for print string literals
///
/// This pass runs after macro expansion to:
/// 1. Find all `__print_internal__` calls
/// 2. Collect unique string literals
/// 3. Generate LLVM global declarations for each string
/// 4. Replace `__print_internal__` calls with proper printf calls
struct StringCollector {
    /// Map from string content to global name
    strings: HashMap<String, String>,
    /// Counter for generating unique names
    counter: usize,
}

impl StringCollector {
    fn new() -> Self {
        Self {
            strings: HashMap::new(),
            counter: 0,
        }
    }

    /// Process a list of values, collecting strings and transforming print calls
    fn process(&mut self, values: Vec<Value>) -> Vec<Value> {
        // First pass: collect all strings and transform __print_internal__ calls
        let transformed: Vec<Value> = values
            .into_iter()
            .map(|v| self.transform_value(v))
            .collect();

        // If no strings were collected, just return the transformed values
        if self.strings.is_empty() {
            return transformed;
        }

        // Generate globals and prepend them
        let mut result = self.generate_globals();
        result.extend(transformed);
        result
    }

    /// Transform a value, replacing __print_internal__ calls
    fn transform_value(&mut self, value: Value) -> Value {
        match value {
            Value::List(items) => {
                // Check if this is a __print_internal__ call
                if let Some(Value::Symbol(sym)) = items.first() {
                    if sym.name == "__print_internal__" {
                        return self.transform_print_call(&items);
                    }
                }
                // Recursively transform children
                Value::List(items.into_iter().map(|v| self.transform_value(v)).collect())
            }
            Value::Vector(items) => {
                Value::Vector(items.into_iter().map(|v| self.transform_value(v)).collect())
            }
            Value::Map(map) => {
                Value::Map(
                    map.into_iter()
                        .map(|(k, v)| (k, self.transform_value(v)))
                        .collect(),
                )
            }
            // Atoms pass through unchanged
            other => other,
        }
    }

    /// Transform a __print_internal__ call to a puts/printf call with addressof
    fn transform_print_call(&mut self, items: &[Value]) -> Value {
        // items[0] is __print_internal__
        // items[1] is the format string
        // items[2..] are additional arguments

        let format_str = match items.get(1) {
            Some(Value::String(s)) => s.clone(),
            _ => return Value::List(items.to_vec()), // Pass through if malformed
        };

        let has_format_args = items.len() > 2;

        // Get or create a global name for this string
        let global_name = self.get_or_create_global(&format_str);

        let ptr_name = format!("_print_ptr_{}", self.counter - 1);

        // Build addressof expression
        let addressof = Value::List(vec![
            Value::symbol("llvm.mlir.addressof"),
            Value::Map(
                [
                    ("global_name".to_string(), Value::symbol(&format!("@{}", global_name))),
                    ("result".to_string(), Value::symbol("!llvm.ptr")),
                ]
                .into_iter()
                .collect(),
            ),
        ]);

        if has_format_args {
            // For strings with format args, we need to call printf with vararg support
            // Generate llvm.call directly with vararg attribute
            // (llvm.call {:callee @printf :vararg (-> [!llvm.ptr ...] [i32]) :result i32} args...)
            let mut attrs = std::collections::HashMap::new();
            attrs.insert("callee".to_string(), Value::symbol("@printf"));
            attrs.insert(
                "vararg".to_string(),
                Value::List(vec![
                    Value::symbol("->"),
                    Value::Vector(vec![Value::symbol("!llvm.ptr"), Value::symbol("...")]),
                    Value::Vector(vec![Value::symbol("i32")]),
                ]),
            );
            attrs.insert("result".to_string(), Value::symbol("i32"));

            let mut printf_call = vec![Value::symbol("llvm.call"), Value::Map(attrs), Value::symbol(&ptr_name)];
            // Add additional arguments (items[2..])
            printf_call.extend(items[2..].iter().cloned());

            Value::List(vec![
                Value::symbol("let"),
                Value::Vector(vec![
                    Value::symbol(&ptr_name),
                    addressof,
                ]),
                Value::List(printf_call),
            ])
        } else {
            // For simple strings without format args, still use llvm.call with vararg
            // since printf is declared as vararg with llvm.func
            let mut attrs = std::collections::HashMap::new();
            attrs.insert("callee".to_string(), Value::symbol("@printf"));
            attrs.insert(
                "vararg".to_string(),
                Value::List(vec![
                    Value::symbol("->"),
                    Value::Vector(vec![Value::symbol("!llvm.ptr"), Value::symbol("...")]),
                    Value::Vector(vec![Value::symbol("i32")]),
                ]),
            );
            attrs.insert("result".to_string(), Value::symbol("i32"));

            let printf_call = vec![Value::symbol("llvm.call"), Value::Map(attrs), Value::symbol(&ptr_name)];

            Value::List(vec![
                Value::symbol("let"),
                Value::Vector(vec![
                    Value::symbol(&ptr_name),
                    addressof,
                ]),
                Value::List(printf_call),
            ])
        }
    }

    /// Get or create a global name for a string
    fn get_or_create_global(&mut self, s: &str) -> String {
        if let Some(name) = self.strings.get(s) {
            return name.clone();
        }

        let name = format!("_print_str_{}", self.counter);
        self.counter += 1;
        self.strings.insert(s.to_string(), name.clone());
        name
    }

    /// Generate LLVM global declarations for all collected strings
    fn generate_globals(&self) -> Vec<Value> {
        let mut globals = Vec::new();

        // Sort by name for deterministic output
        let mut sorted: Vec<_> = self.strings.iter().collect();
        sorted.sort_by_key(|(_, name)| *name);

        for (content, name) in sorted {
            // Need to add null terminator for C strings
            let content_with_null = format!("{}\0", content);
            let len = content_with_null.len();

            // Generate:
            // (llvm.mlir.global {:sym_name "name" :linkage 0 :global_type !llvm.array<len x i8> :constant true}
            //   (region (block []
            //     (def s (llvm.mlir.constant {:value "content\0" :result !llvm.array<len x i8>}))
            //     (llvm.return s))))

            let array_type = format!("!llvm.array<{} x i8>", len);

            let global = Value::List(vec![
                Value::symbol("llvm.mlir.global"),
                Value::Map(
                    [
                        ("sym_name".to_string(), Value::String(name.clone())),
                        ("linkage".to_string(), Value::Number(0.0)), // internal linkage
                        ("global_type".to_string(), Value::symbol(&array_type)),
                        ("constant".to_string(), Value::Boolean(true)),
                    ]
                    .into_iter()
                    .collect(),
                ),
                Value::List(vec![
                    Value::symbol("region"),
                    Value::List(vec![
                        Value::symbol("block"),
                        Value::Vector(vec![]),
                        Value::List(vec![
                            Value::symbol("def"),
                            Value::symbol("_str_val"),
                            Value::List(vec![
                                Value::symbol("llvm.mlir.constant"),
                                Value::Map(
                                    [
                                        // Include null terminator in the string value
                                        ("value".to_string(), Value::String(content_with_null.clone())),
                                        ("result".to_string(), Value::symbol(&array_type)),
                                    ]
                                    .into_iter()
                                    .collect(),
                                ),
                            ]),
                        ]),
                        Value::List(vec![
                            Value::symbol("llvm.return"),
                            Value::symbol("_str_val"),
                        ]),
                    ]),
                ]),
            ]);

            globals.push(global);
        }

        globals
    }
}

impl DynamicMacroContext {

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

        // Post-process: collect print strings and generate LLVM globals
        let mut collector = StringCollector::new();
        result = collector.process(result);

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
