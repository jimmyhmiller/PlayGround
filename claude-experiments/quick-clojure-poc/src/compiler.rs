use crate::arm_codegen::Arm64CodeGen;
use crate::clojure_ast::{CatchClause, Expr};
use crate::gc;
use crate::gc_runtime::GCRuntime;
use crate::ir::{Condition, Instruction, IrBuilder, IrValue, Label};
use crate::trampoline::Trampoline;
use crate::value::Value;
use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::sync::Arc;

/// Context for loop/recur
#[derive(Clone)]
struct LoopContext {
    label: Label,
    binding_registers: Vec<IrValue>,
    /// Local slots for fn parameters - recur must store to these too
    /// None for loop bindings (register-only), Some(slot) for fn params
    local_slots: Vec<Option<usize>>,
}

/// Variable binding - can be a register or a local slot on the stack
/// Following Beagle's pattern: function parameters are stored to local slots
#[derive(Clone, Copy, Debug)]
enum Binding {
    /// Value is in a register (used for let bindings, closure values)
    Register(IrValue),
    /// Value is in a local slot on the stack (used for function parameters)
    /// When accessed, emit LoadLocal to load into a fresh register
    LocalSlot(usize),
}

/// Type information for deftype
#[derive(Clone, Debug)]
struct TypeInfo {
    type_id: usize,
    fields: Vec<(String, bool)>, // (name, is_mutable)
}

/// Clojure to IR compiler
///
/// Compiles Clojure AST to our IR, which is then compiled to ARM64.
pub struct Compiler {
    /// Runtime with heap allocator
    /// SAFETY: Must only be accessed from one thread at a time
    runtime: Arc<UnsafeCell<GCRuntime>>,

    /// Current namespace (HEAP POINTER, not string!)
    current_namespace_ptr: usize,

    /// Namespace registry: name → heap pointer
    namespace_registry: HashMap<String, usize>,

    /// Used namespaces per namespace: namespace → list of used namespaces
    used_namespaces: HashMap<String, Vec<String>>,

    /// IR builder
    pub builder: IrBuilder,

    /// Local variable scopes (for let bindings and parameters)
    /// Each scope maps variable name → Binding (register or local slot)
    /// Stack of scopes allows for nested lets
    local_scopes: Vec<HashMap<String, Binding>>,

    /// Stack of loop contexts for recur
    loop_contexts: Vec<LoopContext>,

    /// Compiled function registry: function_id → code_pointer
    /// Used to track compiled nested functions
    #[allow(dead_code)]
    function_registry: HashMap<usize, usize>,

    /// Next function ID for tracking nested functions
    #[allow(dead_code)]
    next_function_id: usize,

    /// Compiled function IRs for display (name, instructions)
    /// Cleared after each top-level compile
    compiled_function_irs: Vec<(Option<String>, Vec<Instruction>)>,

    /// Type registry for deftype: type_name → TypeInfo
    type_registry: HashMap<String, TypeInfo>,
}

impl Compiler {
    pub fn new(runtime: Arc<UnsafeCell<GCRuntime>>) -> Self {
        // SAFETY: Called during initialization, no concurrent access
        let (core_ns_ptr, user_ns_ptr) = unsafe {
            let rt = &mut *runtime.get();

            // Bootstrap namespaces
            let core_ns_ptr = rt.allocate_namespace("clojure.core").unwrap();
            let user_ns_ptr = rt.allocate_namespace("user").unwrap();

            // Register as GC roots
            rt.add_namespace_root("clojure.core".to_string(), core_ns_ptr);
            rt.add_namespace_root("user".to_string(), user_ns_ptr);

            // Bootstrap *ns* var in clojure.core
            // Initially set to user namespace (since that's the default starting namespace)
            let (ns_var_ptr, symbol_ptr) =
                rt.allocate_var(core_ns_ptr, "*ns*", user_ns_ptr).unwrap();
            let core_ns_ptr = rt
                .namespace_add_binding_with_symbol_ptr(core_ns_ptr, "*ns*", ns_var_ptr, symbol_ptr)
                .unwrap();

            // Mark *ns* as dynamic so it can be rebound later
            rt.mark_var_dynamic(ns_var_ptr);

            // Update namespace root after adding *ns* binding (may have reallocated)
            rt.update_namespace_root("clojure.core", core_ns_ptr);

            // Bootstrap builtin functions as Vars in clojure.core
            let core_ns_ptr = rt.bootstrap_builtins(core_ns_ptr).unwrap();

            // Update namespace root after bootstrap_builtins (may have reallocated many times)
            rt.update_namespace_root("clojure.core", core_ns_ptr);

            (core_ns_ptr, user_ns_ptr)
        };

        let mut namespace_registry = HashMap::new();
        namespace_registry.insert("clojure.core".to_string(), core_ns_ptr);
        namespace_registry.insert("user".to_string(), user_ns_ptr);

        let mut used_namespaces = HashMap::new();
        // Every namespace implicitly uses clojure.core
        used_namespaces.insert("user".to_string(), vec!["clojure.core".to_string()]);

        Compiler {
            runtime,
            current_namespace_ptr: user_ns_ptr,
            namespace_registry,
            used_namespaces,
            builder: IrBuilder::new(),
            local_scopes: Vec::new(),
            loop_contexts: Vec::new(),
            function_registry: HashMap::new(),
            next_function_id: 0,
            compiled_function_irs: Vec::new(),
            type_registry: HashMap::new(),
        }
    }

    /// Compile an expression and return the register containing the result
    pub fn compile(&mut self, expr: &Expr) -> Result<IrValue, String> {
        match expr {
            Expr::Literal(value) => self.compile_literal(value),
            Expr::Var { namespace, name } => self.compile_var(namespace, name),
            Expr::Def {
                name,
                value,
                metadata,
            } => self.compile_def(name, value, metadata),
            Expr::Set { var, value } => self.compile_set(var, value),
            Expr::Ns { name } => self.compile_ns(name),
            Expr::Use { namespace } => self.compile_use(namespace),
            Expr::If { test, then, else_ } => self.compile_if(test, then, else_),
            Expr::Do { exprs } => self.compile_do(exprs),
            Expr::Let { bindings, body } => self.compile_let(bindings, body),
            Expr::Loop { bindings, body } => self.compile_loop(bindings, body),
            Expr::Recur { args } => self.compile_recur(args),
            Expr::Fn { name, arities } => self.compile_fn(name, arities),
            Expr::Binding { bindings, body } => self.compile_binding(bindings, body),
            Expr::Call { func, args } => self.compile_call(func, args),
            Expr::Quote(value) => self.compile_literal(value),
            Expr::VarRef { namespace, name } => self.compile_var_ref(namespace, name),
            Expr::DefType { name, fields } => self.compile_deftype(name, fields),
            Expr::TypeConstruct { type_name, args } => self.compile_type_construct(type_name, args),
            Expr::FieldAccess { field, object } => self.compile_field_access(field, object),
            Expr::FieldSet {
                field,
                object,
                value,
            } => self.compile_field_set(field, object, value),
            Expr::MethodCall {
                method,
                object,
                args,
            } => self.compile_method_call(method, object, args),
            Expr::Throw { exception } => self.compile_throw(exception),
            Expr::Try {
                body,
                catches,
                finally,
            } => self.compile_try(body, catches, finally),
            // Protocol system - will be implemented in Phase 4
            Expr::DefProtocol { name, methods } => self.compile_defprotocol(name, methods),
            Expr::ExtendType {
                type_name,
                implementations,
            } => self.compile_extend_type(type_name, implementations),
            Expr::ProtocolCall { method_name, args } => {
                self.compile_protocol_call(method_name, args)
            }
            Expr::Debugger { expr } => self.compile_debugger(expr),
            Expr::TopLevelDo { .. } => {
                // TopLevelDo should only appear at the top level and is handled by compile_toplevel
                Err("TopLevelDo encountered in nested context - this should not happen".to_string())
            }
        }
    }

    /// Compile a top-level expression (emits Ret instruction)
    /// For TopLevelDo, this processes forms sequentially, executing each before
    /// analyzing the next (required for macros defined and used in same do block).
    pub fn compile_toplevel(&mut self, expr: &Expr) -> Result<IrValue, String> {
        // Handle TopLevelDo specially - compile and execute each form in sequence
        if let Expr::TopLevelDo { forms } = expr {
            return self.compile_toplevel_do(forms);
        }

        // Reset builder for each top-level expression to start with fresh temp registers
        self.builder.reset();
        let result = self.compile(expr)?;
        self.builder.emit(Instruction::Ret(result));
        Ok(result)
    }

    /// Compile and execute a top-level do block, processing forms sequentially.
    /// This is needed so that macros defined in earlier forms can be used in later forms.
    fn compile_toplevel_do(&mut self, forms: &[usize]) -> Result<IrValue, String> {
        use crate::arm_codegen::Arm64CodeGen;
        use crate::clojure_ast::analyze_tagged;
        use crate::trampoline::Trampoline;

        // Register all forms as temporary GC roots before processing.
        // This prevents forms from being collected when GC runs during analysis
        // of earlier forms.
        let root_ids: Vec<usize> = unsafe {
            let rt = &mut *self.runtime.get();
            forms
                .iter()
                .map(|&tagged| rt.register_temporary_root(tagged))
                .collect()
        };

        let mut last_result = IrValue::Null;

        for (i, &tagged) in forms.iter().enumerate() {
            let is_last = i == forms.len() - 1;

            // Analyze this form
            // SAFETY: We need mutable access to runtime for analysis
            let ast = unsafe {
                let rt = &mut *self.runtime.get();
                analyze_tagged(rt, tagged)?
            };

            // Compile this form
            self.builder.reset();
            let result = self.compile(&ast)?;
            self.builder.emit(Instruction::Ret(result));

            let instructions = self.take_instructions();
            let num_locals = self.builder.num_locals;

            // Generate machine code
            let compiled = Arm64CodeGen::compile_function(&instructions, num_locals, 0)
                .map_err(|e| format!("Codegen error in toplevel do: {}", e))?;

            // Register stack maps BEFORE execution - critical for GC safety
            unsafe {
                let rt = &mut *self.runtime.get();
                for (pc, stack_size) in &compiled.stack_map {
                    rt.add_stack_map_entry(
                        *pc,
                        gc::StackMapDetails {
                            function_name: None,
                            number_of_locals: compiled.num_stack_slots,
                            current_stack_size: *stack_size,
                            max_stack_size: *stack_size,
                        },
                    );
                }
            }

            // Execute this form
            let trampoline = Trampoline::new(64 * 1024);
            let exec_result = unsafe { trampoline.execute(compiled.code_ptr as *const u8) };

            // If this is the last form, we need to return its result
            // For intermediate forms, we just need them to execute (for side effects like def)
            if is_last {
                // For the last form, we need to reload the result as a constant
                // since we've already executed it
                last_result = IrValue::TaggedConstant(exec_result as isize);
            }
        }

        // Emit final return with the result of the last expression
        self.builder.reset();
        self.builder.emit(Instruction::Ret(last_result));

        // Unregister all temporary roots
        unsafe {
            let rt = &mut *self.runtime.get();
            for root_id in root_ids {
                rt.unregister_temporary_root(root_id);
            }
        }

        Ok(last_result)
    }

    fn compile_literal(&mut self, value: &Value) -> Result<IrValue, String> {
        // OPTIMIZATION: Return constants directly without loading into registers.
        // This prevents long-lived registers for literals in nested expressions.
        // The codegen will load constants into temp registers when needed.
        match value {
            Value::Nil => {
                return Ok(IrValue::Null);
            }
            Value::Bool(true) => {
                // true: (1 << 3) | 0b011 = 11
                return Ok(IrValue::TaggedConstant(11));
            }
            Value::Bool(false) => {
                // false: (0 << 3) | 0b011 = 3
                return Ok(IrValue::TaggedConstant(3));
            }
            Value::Int(i) => {
                // Integers: value << 3 | 0b000 (tag 0)
                let tagged = (*i as isize) << 3;
                return Ok(IrValue::TaggedConstant(tagged));
            }
            _ => {} // Fall through to allocate a register for heap values
        }

        // Heap-allocated values (floats, strings, keywords) need registers
        let result = self.builder.new_register();

        match value {
            Value::Nil | Value::Bool(_) | Value::Int(_) => {
                unreachable!("Handled above")
            }
            Value::Float(f) => {
                // Floats are heap-allocated to preserve full precision
                // SAFETY: Single-threaded REPL
                let rt = unsafe { &mut *self.runtime.get() };
                let float_ptr = rt
                    .allocate_float(*f)
                    .map_err(|e| format!("Failed to allocate float: {}", e))?;

                // Load the tagged pointer as a constant
                self.builder.emit(Instruction::LoadConstant(
                    result,
                    IrValue::TaggedConstant(float_ptr as isize),
                ));
            }
            Value::String(s) => {
                // Allocate string at compile time as a rooted constant
                // This ensures the string survives GC even when not stored in a var
                // SAFETY: Single-threaded REPL
                let rt = unsafe { &mut *self.runtime.get() };
                let str_ptr = rt
                    .allocate_string_constant(s)
                    .map_err(|e| format!("Failed to allocate string: {}", e))?;

                // Load the tagged pointer as a constant
                self.builder.emit(Instruction::LoadConstant(
                    result,
                    IrValue::TaggedConstant(str_ptr as isize),
                ));
            }
            Value::Keyword(text) => {
                // Add keyword to constants table at compile time
                // SAFETY: Single-threaded REPL
                let rt = unsafe { &mut *self.runtime.get() };
                let keyword_index = rt.add_keyword(text.clone());

                // Call load-keyword builtin - actual allocation happens at runtime
                let index_reg = self
                    .builder
                    .assign_new(IrValue::TaggedConstant((keyword_index << 3) as isize));
                return self.call_builtin("load-keyword", vec![index_reg]);
            }
            Value::Symbol(text) => {
                // Allocate symbol at compile time as a ReaderSymbol
                // Parse namespace/name from the symbol text
                // SAFETY: Single-threaded REPL
                let rt = unsafe { &mut *self.runtime.get() };

                let (namespace, name) = if let Some(slash_pos) = text.find('/') {
                    (Some(&text[..slash_pos]), &text[slash_pos + 1..])
                } else {
                    (None, text.as_str())
                };

                let sym_ptr = rt
                    .allocate_reader_symbol(namespace, name)
                    .map_err(|e| format!("Failed to allocate symbol: {}", e))?;

                // Load the tagged pointer as a constant
                self.builder.emit(Instruction::LoadConstant(
                    result,
                    IrValue::TaggedConstant(sym_ptr as isize),
                ));
            }
            Value::Vector(elements) => {
                // Compile vector literal as call to (vector elem1 elem2 ...)
                // Convert each element to an Expr and compile as a vector call
                let elem_exprs: Vec<Expr> =
                    elements.iter().map(|v| Expr::Literal(v.clone())).collect();
                return self.compile_call(
                    &Expr::Var {
                        namespace: Some("clojure.core".to_string()),
                        name: "vector".to_string(),
                    },
                    &elem_exprs,
                );
            }
            Value::Map(pairs) => {
                // Compile map literal as call to (hash-map k1 v1 k2 v2 ...)
                // Flatten the map into key-value pairs
                let mut kv_exprs: Vec<Expr> = Vec::new();
                for (k, v) in pairs.iter() {
                    kv_exprs.push(Expr::Literal(k.clone()));
                    kv_exprs.push(Expr::Literal(v.clone()));
                }
                return self.compile_call(
                    &Expr::Var {
                        namespace: Some("clojure.core".to_string()),
                        name: "hash-map".to_string(),
                    },
                    &kv_exprs,
                );
            }
            Value::List(elements) => {
                // Compile list literal as call to (list elem1 elem2 ...)
                let elem_exprs: Vec<Expr> =
                    elements.iter().map(|v| Expr::Literal(v.clone())).collect();
                return self.compile_call(
                    &Expr::Var {
                        namespace: Some("clojure.core".to_string()),
                        name: "list".to_string(),
                    },
                    &elem_exprs,
                );
            }
            Value::Set(elements) => {
                // Compile set literal as call to (hash-set elem1 elem2 ...)
                let elem_exprs: Vec<Expr> =
                    elements.iter().map(|v| Expr::Literal(v.clone())).collect();
                return self.compile_call(
                    &Expr::Var {
                        namespace: Some("clojure.core".to_string()),
                        name: "hash-set".to_string(),
                    },
                    &elem_exprs,
                );
            }
            _ => {
                return Err(format!("Literal type not yet supported: {:?}", value));
            }
        }

        Ok(result)
    }

    fn compile_var(&mut self, namespace: &Option<String>, name: &str) -> Result<IrValue, String> {
        // First, check if this is a local variable (from let)
        // Only unqualified names can be locals
        if namespace.is_none()
            && let Some(register) = self.lookup_local(name)
        {
            // Local variable - just return the register directly
            return Ok(register);
        }

        // Not a local - emit runtime symbol lookup
        // This enables forward references: vars are looked up at runtime, not compile time

        // SAFETY: Single-threaded REPL - no concurrent access during compilation
        let rt = unsafe { &mut *self.runtime.get() };

        // Determine namespace name
        let ns_name = if let Some(ns) = namespace {
            // Verify namespace exists
            if !self.namespace_registry.contains_key(ns) {
                return Err(format!("Undefined namespace: {}", ns));
            }
            ns.clone()
        } else {
            // Use current namespace
            rt.namespace_name(self.current_namespace_ptr)
        };

        // Intern symbols for runtime lookup
        let ns_symbol_id = rt.intern_symbol(&ns_name);
        let name_symbol_id = rt.intern_symbol(name);

        // Check if var exists and is dynamic at compile time
        // If var doesn't exist yet (forward reference), we can't know - assume non-dynamic
        let is_dynamic = if let Some(&ns_ptr) = self.namespace_registry.get(&ns_name) {
            if let Some(var_ptr) = rt.namespace_lookup(ns_ptr, name) {
                rt.is_var_dynamic(var_ptr)
            } else {
                false // Forward reference - assume non-dynamic
            }
        } else {
            false
        };

        // Call builtin function to load var value
        let builtin_name = if is_dynamic {
            "load-var-by-symbol-dynamic"
        } else {
            "load-var-by-symbol"
        };

        let ns_sym_reg = self
            .builder
            .assign_new(IrValue::TaggedConstant((ns_symbol_id << 3) as isize));
        let name_sym_reg = self
            .builder
            .assign_new(IrValue::TaggedConstant((name_symbol_id << 3) as isize));

        self.call_builtin(builtin_name, vec![ns_sym_reg, name_sym_reg])
    }

    /// Check if a symbol is a built-in function
    fn is_builtin(&self, name: &str) -> bool {
        matches!(
            name,
            "prim-add" | "prim-sub" | "prim-mul" | "prim-div" | "<" | ">" | "<=" | ">=" | "=" | "__gc" |
                 "bit-and" | "bit-or" | "bit-xor" | "bit-not" |
                 "bit-shift-left" | "bit-shift-right" | "unsigned-bit-shift-right" |
                 "bit-shift-right-zero-fill" |  // CLJS alias for unsigned-bit-shift-right
                 "nil?" | "number?" | "string?" | "fn?" | "identical?" |
                 "instance?" | "satisfies?" | "keyword?" | "hash-primitive" |
                 "cons?" | "cons-first" | "cons-rest" |
                 "_println" | "_print" | "_newline" | "_print-space" |
                 "__make_reader_symbol_1" | "__make_reader_symbol_2" |
                 "__make_reader_list_0" | "__make_reader_list_from_vec" |
                 "__make_reader_vector_0" | "__make_reader_vector_from_list" |
                 "__make_reader_map_0" |
                 "__reader_cons" | "__reader_list_1" | "__reader_list_2" |
                 "__reader_list_3" | "__reader_list_4" | "__reader_list_5" |
                 // Reader primitive operations for protocol implementations
                 "__reader_list_first" | "__reader_list_rest" | "__reader_list_count" |
                 "__reader_list_nth" | "__reader_list_conj" |
                 "__reader_vector_first" | "__reader_vector_rest" | "__reader_vector_count" |
                 "__reader_vector_nth" | "__reader_vector_conj" |
                 // Reader type predicates
                 "__reader_list?" | "__reader_vector?" | "__reader_map?" | "__reader_symbol?" |
                 // New builtins for defmacro support
                 "__is_string" | "__is_symbol" | "__set_macro!" |
                 "__symbol_1" | "__symbol_2" |
                 "__gensym_0" | "__gensym_1" |
                 // Array operations
                 "make-array" | "aget" | "aset" | "aset!" | "alength" | "aclone" |
                 // Core function builtins
                 "__keyword_name" | "__apply" | "__str_concat" |
                 // Reader set operations
                 "__reader_set_count" | "__reader_set_contains" | "__reader_set_conj" | "__reader_set_get" |
                 "__is_reader_set"
        )
    }

    /// Compile (var symbol) - returns the Var object itself, not its value
    fn compile_var_ref(
        &mut self,
        namespace: &Option<String>,
        name: &str,
    ) -> Result<IrValue, String> {
        // Look up the var pointer (same logic as compile_var, but don't dereference)
        let var_ptr = self.lookup_var_pointer(namespace, name)?;

        // Return the var pointer tagged as a pointer (tag 010 = shifted left 3 + 2)
        // Actually, vars are already tagged when stored, so we return them as-is
        let result = self.builder.new_register();
        self.builder.emit(Instruction::LoadConstant(
            result,
            IrValue::TaggedConstant(var_ptr as isize),
        ));

        Ok(result)
    }

    /// Compile (deftype* TypeName [field1 ^:mutable field2 ...])
    /// Registers the type and returns nil
    fn compile_deftype(
        &mut self,
        name: &str,
        fields: &[crate::clojure_ast::FieldDef],
    ) -> Result<IrValue, String> {
        // Register the type with the runtime using fully qualified name
        let current_ns = self.get_current_namespace();
        let qualified_name = format!("{}/{}", current_ns, name);

        // Extract just the field names for runtime registration
        let field_names: Vec<String> = fields.iter().map(|f| f.name.clone()).collect();

        // Get mutable access to runtime to register type
        let type_id = {
            let rt = unsafe { &mut *self.runtime.get() };
            rt.register_type(qualified_name.clone(), field_names)
        };

        // Store in compiler's type registry with FULLY QUALIFIED name
        // This prevents namespace collisions and enables qualified constructor calls
        // Also track mutability for each field
        let type_info = TypeInfo {
            type_id,
            fields: fields.iter().map(|f| (f.name.clone(), f.mutable)).collect(),
        };
        self.type_registry.insert(qualified_name, type_info);

        // deftype* returns nil
        let result = self.builder.new_register();
        self.builder
            .emit(Instruction::LoadConstant(result, IrValue::Null));
        Ok(result)
    }

    /// Compile (TypeName. arg1 arg2 ...) - constructor call
    ///
    /// REFACTORED: Instead of emitting a single MakeType instruction, we emit
    /// a sequence of primitive IR instructions following the Beagle pattern:
    /// 1. ExternalCall to allocate heap object (returns untagged pointer)
    /// 2. HeapStore for each field value
    /// 3. Tag the pointer with HeapObject tag
    fn compile_type_construct(
        &mut self,
        type_name: &str,
        args: &[Expr],
    ) -> Result<IrValue, String> {
        // Resolve type name to fully qualified name
        let qualified_name = if type_name.contains('/') {
            type_name.to_string()
        } else {
            format!("{}/{}", self.get_current_namespace(), type_name)
        };

        // Look up the type by fully qualified name
        let type_info = self
            .type_registry
            .get(&qualified_name)
            .cloned()
            .ok_or_else(|| format!("Unknown type: {}", qualified_name))?;

        // Check arg count matches field count
        if args.len() != type_info.fields.len() {
            return Err(format!(
                "Type {} requires {} arguments, got {}",
                type_name,
                type_info.fields.len(),
                args.len()
            ));
        }

        // Compile all arguments and ensure they're in registers
        let mut field_values = Vec::new();
        for arg in args {
            let val = self.compile(arg)?;
            let reg = self.ensure_register(val);
            field_values.push(reg);
        }

        // Step 1: Allocate heap object via trampoline
        // builtin_allocate_type_object_raw(frame_pointer, gc_return_addr, type_id, field_count) -> untagged_ptr
        let builtin_addr = crate::trampoline::builtin_allocate_type_object_raw as usize;
        let raw_ptr = self.builder.new_register();

        // Load type_id and field_count as raw constants (untagged values)
        let type_id_reg = self
            .builder
            .assign_new(IrValue::RawConstant(type_info.type_id as i64));
        let field_count_reg = self
            .builder
            .assign_new(IrValue::RawConstant(field_values.len() as i64));

        // Pass FramePointer (x29) for GC stack walking. gc_return_addr is computed internally.
        self.builder.emit(Instruction::ExternalCall(
            raw_ptr,
            builtin_addr,
            vec![IrValue::FramePointer, type_id_reg, field_count_reg],
        ));

        // Step 2: Write field values using HeapStore
        // Field 0 is at offset 1 (after 8-byte header), field 1 at offset 2, etc.
        for (i, field_val) in field_values.iter().enumerate() {
            let offset = (i + 1) as i32;
            self.builder
                .emit(Instruction::HeapStore(raw_ptr, offset, *field_val));
        }

        // Step 3: Tag the pointer with HeapObject tag (0b110)
        // Pass the tag directly as a constant (don't allocate a register for it)
        let result = self.builder.new_register();
        self.builder.emit(Instruction::Tag(
            result,
            raw_ptr,
            IrValue::RawConstant(0b110),
        ));

        Ok(result)
    }

    /// Compile (.-field obj) - field access
    ///
    /// REFACTORED: Now uses ExternalCall with pre-interned symbol ID instead of
    /// the LoadTypeField instruction that embedded field names on the stack.
    fn compile_field_access(&mut self, field: &str, object: &Expr) -> Result<IrValue, String> {
        // Compile the object and ensure it's in a register
        let obj_value = self.compile(object)?;
        let obj_reg = self.ensure_register(obj_value);

        // Intern the field name as a symbol at compile time
        let rt = unsafe { &mut *self.runtime.get() };
        let field_symbol_id = rt.intern_symbol(field);

        // Emit ExternalCall to builtin_load_type_field_by_symbol(obj, symbol_id)
        let builtin_addr = crate::trampoline::builtin_load_type_field_by_symbol as usize;
        let symbol_id_reg = self
            .builder
            .assign_new(IrValue::RawConstant(field_symbol_id as i64));

        let result = self.builder.new_register();
        self.builder.emit(Instruction::ExternalCall(
            result,
            builtin_addr,
            vec![obj_reg, symbol_id_reg],
        ));

        Ok(result)
    }

    /// Compile (set! (.-field obj) value) - field assignment
    /// Requires the field to be declared as ^:mutable in the deftype
    ///
    /// REFACTORED: Now uses ExternalCall with pre-interned symbol ID instead of
    /// the StoreTypeField instruction that embedded field names on the stack.
    fn compile_field_set(
        &mut self,
        field: &str,
        object: &Expr,
        value: &Expr,
    ) -> Result<IrValue, String> {
        // 1. Compile the object and ensure it's in a register
        let obj_value = self.compile(object)?;
        let obj_reg = self.ensure_register(obj_value);

        // 2. Compile the new value and ensure it's in a register
        let new_value = self.compile(value)?;
        let new_value_reg = self.ensure_register(new_value);

        // 3. Emit GcAddRoot as ExternalCall for write barrier (BEFORE the store)
        // This is critical for generational GC correctness
        let gc_builtin_addr = crate::trampoline::builtin_gc_add_root as usize;
        let gc_result = self.builder.new_register();
        self.builder.emit(Instruction::ExternalCall(
            gc_result,
            gc_builtin_addr,
            vec![obj_reg],
        ));

        // 4. Intern the field name as a symbol at compile time
        let rt = unsafe { &mut *self.runtime.get() };
        let field_symbol_id = rt.intern_symbol(field);

        // 5. Emit ExternalCall to builtin_store_type_field_by_symbol(obj, symbol_id, value)
        let store_builtin_addr =
            crate::trampoline::builtin_store_type_field_by_symbol as usize;
        let symbol_id_reg = self
            .builder
            .assign_new(IrValue::RawConstant(field_symbol_id as i64));

        let result = self.builder.new_register();
        self.builder.emit(Instruction::ExternalCall(
            result,
            store_builtin_addr,
            vec![obj_reg, symbol_id_reg, new_value_reg],
        ));

        // 6. set! returns the stored value
        Ok(result)
    }

    /// Compile (.method obj arg1 arg2 ...) - method call on deftype
    /// Uses protocol lookup to find the method implementation
    fn compile_method_call(
        &mut self,
        method: &str,
        object: &Expr,
        args: &[Expr],
    ) -> Result<IrValue, String> {
        // Compile the object (first argument to method)
        let obj_value = self.compile(object)?;
        let obj_reg = self.ensure_register(obj_value);

        // Compile additional arguments
        let mut arg_regs = vec![obj_reg];
        for arg in args {
            let arg_value = self.compile(arg)?;
            arg_regs.push(self.ensure_register(arg_value));
        }

        // Leak method name string for trampoline (static lifetime)
        let method_bytes = method.as_bytes();
        let method_ptr = Box::leak(method_bytes.to_vec().into_boxed_slice()).as_ptr() as usize;
        let method_len = method.len();

        // Step 1: Call builtin_protocol_lookup(target, method_ptr, method_len) -> fn_ptr
        let fn_ptr = self.builder.new_register();
        let builtin_addr = crate::trampoline::builtin_protocol_lookup as usize;
        self.builder.emit(Instruction::ExternalCall(
            fn_ptr,
            builtin_addr,
            vec![
                arg_regs[0],                              // target object
                IrValue::RawConstant(method_ptr as i64),  // method_name_ptr
                IrValue::RawConstant(method_len as i64),  // method_name_len
            ],
        ));

        // Step 2: Call fn_ptr with all args (object + additional args)
        let result = self.builder.new_register();
        self.builder
            .emit(Instruction::Call(result, fn_ptr, arg_regs));

        Ok(result)
    }

    fn compile_throw(&mut self, exception: &Expr) -> Result<IrValue, String> {
        // Compile the exception expression
        let exc_val = self.compile(exception)?;

        // Emit Throw instruction
        self.builder.emit(Instruction::Throw(exc_val));

        // Throw never returns, but return dummy nil for type consistency
        let dummy = self.builder.new_register();
        self.builder
            .emit(Instruction::LoadConstant(dummy, IrValue::Null));
        Ok(dummy)
    }

    fn compile_try(
        &mut self,
        body: &[Expr],
        catches: &[CatchClause],
        finally: &Option<Vec<Expr>>,
    ) -> Result<IrValue, String> {
        // Layout:
        //   exception_local = nil  (allocate local for exception)
        //   push_exception_handler(catch_label, exception_local)
        //   <try body>
        //   result = body_result
        //   pop_exception_handler
        //   <finally if present>
        //   jump after_catch_label
        // catch_label:
        //   <catch body with exception_local bound>
        //   result = catch_result
        //   <finally if present>
        // after_catch_label:
        //   return result

        let result = self.builder.new_register();
        let catch_label = self.builder.new_label();
        let after_catch_label = self.builder.new_label();

        // Allocate a dedicated stack slot for the exception value
        // This is allocated BEFORE register allocation runs, ensuring the
        // stack frame is properly sized in the prologue
        let exception_slot = self.builder.allocate_exception_slot();

        // Register to hold the exception value after loading from stack
        let exception_local = self.builder.new_register();

        // 1. Push exception handler with the pre-allocated slot index
        self.builder.emit(Instruction::PushExceptionHandler(
            catch_label.clone(),
            exception_slot,
        ));

        // 2. Compile try body
        let body_result = self.compile_do(body)?;
        self.builder.emit(Instruction::Assign(result, body_result));

        // 3. Pop handler (normal exit)
        self.builder.emit(Instruction::PopExceptionHandler);

        // 4. Handle finally for normal path
        if let Some(finally_body) = finally {
            self.compile_do(finally_body)?;
        }
        self.builder
            .emit(Instruction::Jump(after_catch_label.clone()));

        // 5. Catch block (throw jumps here with exception stored at stack slot)
        self.builder.emit(Instruction::Label(catch_label.clone()));

        // Load exception from the pre-allocated stack slot into exception_local register
        self.builder.emit(Instruction::LoadExceptionLocal(
            exception_local,
            exception_slot,
        ));

        // Handle catch clauses
        if !catches.is_empty() {
            // For now, just handle the first catch clause (catch all)
            let catch = &catches[0];
            self.push_scope();
            self.bind_local(catch.binding.clone(), exception_local);

            let catch_result = self.compile_do(&catch.body)?;
            self.builder.emit(Instruction::Assign(result, catch_result));

            self.pop_scope();
        } else {
            // No catch - just re-throw
            self.builder.emit(Instruction::Throw(exception_local));
        }

        // 6. Handle finally for catch path
        if let Some(finally_body) = finally {
            self.compile_do(finally_body)?;
        }

        // 7. End
        self.builder.emit(Instruction::Label(after_catch_label));
        Ok(result)
    }

    // ========== Protocol Compilation Methods ==========
    // These are stubs that will be fully implemented in Phase 4

    fn compile_defprotocol(
        &mut self,
        name: &str,
        methods: &[crate::clojure_ast::ProtocolMethodSig],
    ) -> Result<IrValue, String> {
        // Register the protocol in the runtime
        use crate::gc_runtime::ProtocolMethod;

        let protocol_methods: Vec<ProtocolMethod> = methods
            .iter()
            .map(|m| ProtocolMethod {
                name: m.name.clone(),
            })
            .collect();

        // Fully qualify the protocol name
        let full_name = format!("{}/{}", self.get_current_namespace(), name);

        unsafe {
            let rt = &mut *self.runtime.get();
            rt.register_protocol(full_name, protocol_methods);
        }

        // For each method, create a dispatch function and def it in the current namespace
        for method in methods {
            // Synthesize function arities that dispatch to the protocol method
            let fn_arities: Vec<crate::value::FnArity> = method
                .arities
                .iter()
                .map(|params| {
                    // Build args for the protocol call (references to the params)
                    let args: Vec<Expr> = params
                        .iter()
                        .map(|p| Expr::Var {
                            namespace: None,
                            name: p.clone(),
                        })
                        .collect();

                    crate::value::FnArity {
                        params: params.clone(),
                        rest_param: None,
                        body: vec![Expr::ProtocolCall {
                            method_name: method.name.clone(),
                            args,
                        }],
                        pre_conditions: Vec::new(),
                        post_conditions: Vec::new(),
                    }
                })
                .collect();

            // Synthesize the function expression
            let fn_expr = Expr::Fn {
                name: Some(method.name.clone()),
                arities: fn_arities,
            };

            // Synthesize a def for this method
            let def_expr = Expr::Def {
                name: method.name.clone(),
                value: Box::new(fn_expr),
                metadata: None,
            };

            // Compile the def
            self.compile(&def_expr)?;
        }

        // defprotocol returns nil
        let result = self.builder.new_register();
        self.builder
            .emit(Instruction::LoadConstant(result, IrValue::Null));
        Ok(result)
    }

    fn compile_extend_type(
        &mut self,
        type_name: &str,
        implementations: &[crate::clojure_ast::ProtocolImpl],
    ) -> Result<IrValue, String> {
        use std::collections::HashMap;

        // Resolve type_id
        let type_id = self.resolve_type_id(type_name)?;

        // For each protocol implementation
        for impl_ in implementations {
            // Get protocol_id
            let protocol_full_name =
                format!("{}/{}", self.get_current_namespace(), &impl_.protocol_name);
            let protocol_id = unsafe {
                let rt = &*self.runtime.get();
                rt.get_protocol_id(&protocol_full_name)
                    .ok_or_else(|| format!("Unknown protocol: {}", impl_.protocol_name))?
            };

            // Check if this is a marker protocol (no methods in the implementation)
            if impl_.methods.is_empty() {
                // Register marker protocol satisfaction
                let builtin_addr =
                    crate::trampoline::builtin_register_marker_protocol as usize;
                let dummy_result = self.builder.new_register();
                self.builder.emit(Instruction::ExternalCall(
                    dummy_result,
                    builtin_addr,
                    vec![
                        IrValue::RawConstant(type_id as i64),
                        IrValue::RawConstant(protocol_id as i64),
                    ],
                ));
                continue;
            }

            // Group methods by name to handle multi-arity protocol methods
            // Multiple implementations with the same name but different param counts
            // should be combined into a single multi-arity function
            let mut methods_by_name: HashMap<String, Vec<&crate::clojure_ast::ProtocolMethodImpl>> =
                HashMap::new();
            for method in &impl_.methods {
                methods_by_name
                    .entry(method.name.clone())
                    .or_default()
                    .push(method);
            }

            // For each unique method name
            for (method_name, method_impls) in methods_by_name {
                let method_index = unsafe {
                    let rt = &*self.runtime.get();
                    rt.get_protocol_method_index(protocol_id, &method_name)
                        .ok_or_else(|| {
                            format!(
                                "Unknown method {} in protocol {}",
                                method_name, impl_.protocol_name
                            )
                        })?
                };

                // Compile all arities of this method into FnArity structs
                let arities: Vec<crate::value::FnArity> = method_impls
                    .iter()
                    .map(|method| crate::value::FnArity {
                        params: method.params.clone(),
                        rest_param: None,
                        body: method.body.clone(),
                        pre_conditions: vec![],
                        post_conditions: vec![],
                    })
                    .collect();

                // Compile the method(s) using compile_fn logic
                // If there are multiple arities, this creates a multi-arity function
                let fn_value = self.compile_fn(&None, &arities)?;

                // Emit RegisterProtocolMethod as ExternalCall
                // Call builtin_register_protocol_method(type_id, protocol_id, method_index, fn_ptr)
                let builtin_addr =
                    crate::trampoline::builtin_register_protocol_method as usize;
                let dummy_result = self.builder.new_register();
                self.builder.emit(Instruction::ExternalCall(
                    dummy_result,
                    builtin_addr,
                    vec![
                        IrValue::RawConstant(type_id as i64),
                        IrValue::RawConstant(protocol_id as i64),
                        IrValue::RawConstant(method_index as i64),
                        fn_value,
                    ],
                ));
            }
        }

        // Return nil
        let result = self.builder.new_register();
        self.builder
            .emit(Instruction::LoadConstant(result, IrValue::Null));
        Ok(result)
    }

    /// Resolve a type name to its type_id for protocol dispatch
    fn resolve_type_id(&self, type_name: &str) -> Result<usize, String> {
        use crate::gc_runtime::*;

        // Check for registered deftypes FIRST - this allows user-defined types
        // to shadow built-in type names (e.g., deftype PersistentVector)
        let full_name = format!("{}/{}", self.get_current_namespace(), type_name);
        let deftype_id = unsafe {
            let rt = &*self.runtime.get();
            rt.get_type_id(&full_name)
        };

        if let Some(id) = deftype_id {
            return Ok(id + DEFTYPE_ID_OFFSET);
        }

        // Fall back to built-in types
        match type_name {
            "nil" | "Nil" => Ok(TYPE_NIL),
            "Boolean" | "bool" => Ok(TYPE_BOOL),
            "Long" | "Integer" | "int" => Ok(TYPE_INT),
            "Double" | "Float" | "float" => Ok(TYPE_FLOAT),
            "String" => Ok(TYPE_STRING),
            "Keyword" => Ok(TYPE_KEYWORD),
            "Symbol" => Ok(TYPE_SYMBOL),
            "PersistentList" | "List" => Ok(TYPE_LIST),
            "PersistentVector" | "Vector" => Ok(TYPE_VECTOR),
            "PersistentHashMap" | "Map" => Ok(TYPE_MAP),
            "PersistentHashSet" | "Set" => Ok(TYPE_SET),
            "Array" => Ok(TYPE_ARRAY),
            // Reader types - available for extend-type in clojure.core
            // Named with __ prefix to indicate internal/primitive types
            "__ReaderList" => Ok(TYPE_READER_LIST),
            "__ReaderVector" => Ok(TYPE_READER_VECTOR),
            "__ReaderMap" => Ok(TYPE_READER_MAP),
            "__ReaderSymbol" => Ok(TYPE_READER_SYMBOL),
            "__ReaderSet" => Ok(TYPE_READER_SET),
            _ => Err(format!("Unknown type: {}", type_name)),
        }
    }

    fn compile_protocol_call(
        &mut self,
        method_name: &str,
        args: &[Expr],
    ) -> Result<IrValue, String> {
        if args.is_empty() {
            return Err(format!(
                "Protocol method {} requires at least 1 argument",
                method_name
            ));
        }

        // Compile all arguments
        let mut arg_values = Vec::new();
        for arg in args {
            arg_values.push(self.compile(arg)?);
        }

        // Leak method name string for trampoline (static lifetime)
        let method_bytes = method_name.as_bytes();
        let method_ptr = Box::leak(method_bytes.to_vec().into_boxed_slice()).as_ptr() as usize;
        let method_len = method_name.len();

        // Ensure all arg values are in registers (convert constants if needed)
        // This is important: constants in codegen use temp registers which can be
        // clobbered by internal Call computation.
        let arg_values: Vec<_> = arg_values
            .into_iter()
            .map(|v| self.builder.assign_new(v))
            .collect();

        // Step 1: Call builtin_protocol_lookup(target, method_ptr, method_len) -> fn_ptr
        let fn_ptr = self.builder.new_register();
        let builtin_addr = crate::trampoline::builtin_protocol_lookup as usize;
        self.builder.emit(Instruction::ExternalCall(
            fn_ptr,
            builtin_addr,
            vec![
                arg_values[0],                   // target
                IrValue::RawConstant(method_ptr as i64), // method_name_ptr
                IrValue::RawConstant(method_len as i64), // method_name_len
            ],
        ));

        // Step 2: Call fn_ptr with all args (register allocator handles saves)
        let result = self.builder.new_register();
        self.builder
            .emit(Instruction::Call(result, fn_ptr, arg_values));

        Ok(result)
    }

    fn compile_debugger(&mut self, expr: &Expr) -> Result<IrValue, String> {
        // Emit breakpoint instruction before evaluating the expression
        self.builder.emit(Instruction::Breakpoint);
        // Compile and return the expression
        self.compile(expr)
    }

    fn compile_def(
        &mut self,
        name: &str,
        value_expr: &Expr,
        metadata: &Option<im::HashMap<String, Value>>,
    ) -> Result<IrValue, String> {
        // Check if var has ^:dynamic metadata
        let is_dynamic = metadata
            .as_ref()
            .and_then(|m| m.get("dynamic"))
            .map(|v| match v {
                Value::Bool(true) => true,
                Value::Keyword(k) if k == "dynamic" => true,
                _ => false,
            })
            .unwrap_or(false);

        // Check if var has ^:macro metadata
        let is_macro = metadata
            .as_ref()
            .and_then(|m| m.get("macro"))
            .map(|v| match v {
                Value::Bool(true) => true,
                Value::Keyword(k) if k == "macro" => true,
                _ => false,
            })
            .unwrap_or(false);

        // SAFETY: Single-threaded REPL - no concurrent access during compilation
        let rt = unsafe { &mut *self.runtime.get() };

        // Get namespace name and intern symbols for runtime lookup
        let ns_name = rt.namespace_name(self.current_namespace_ptr);
        let ns_symbol_id = rt.intern_symbol(&ns_name);
        let name_symbol_id = rt.intern_symbol(name);

        // Create/lookup the Var BEFORE compiling the value expression.
        // This allows recursive functions to resolve their own name.
        // Check if var already exists
        if let Some(existing_var_ptr) = rt.namespace_lookup(self.current_namespace_ptr, name) {
            // Existing var - mark dynamic/macro if needed
            if is_dynamic {
                rt.mark_var_dynamic(existing_var_ptr);
            }
            if is_macro {
                rt.set_var_macro(existing_var_ptr)
                    .map_err(|e| format!("Failed to mark var as macro: {}", e))?;
            }
        } else {
            // Create new var with nil placeholder (7) - will be overwritten after compilation
            let (new_var_ptr, symbol_ptr) = rt
                .allocate_var(
                    self.current_namespace_ptr,
                    name,
                    7, // nil placeholder
                )
                .unwrap();

            // Mark as dynamic ONLY if metadata indicates it
            if is_dynamic {
                rt.mark_var_dynamic(new_var_ptr);
            }

            // Mark as macro if metadata indicates it
            if is_macro {
                rt.set_var_macro(new_var_ptr)
                    .map_err(|e| format!("Failed to mark var as macro: {}", e))?;
            }

            // Add var to namespace BEFORE compiling value
            // This enables recursive function references
            let new_ns_ptr = rt
                .namespace_add_binding_with_symbol_ptr(
                    self.current_namespace_ptr,
                    name,
                    new_var_ptr,
                    symbol_ptr,
                )
                .unwrap();

            // Update current namespace pointer if it moved
            if new_ns_ptr != self.current_namespace_ptr {
                self.current_namespace_ptr = new_ns_ptr;

                // Update in registry
                let ns_name_updated = rt.namespace_name(new_ns_ptr);
                self.namespace_registry
                    .insert(ns_name_updated.clone(), new_ns_ptr);

                // Update the runtime's namespace root so trampolines can find it
                rt.update_namespace_root(&ns_name_updated, new_ns_ptr);
            }
        }

        // NOW compile the value expression - recursive refs will resolve via load-var-by-symbol builtin
        let value_reg = self.compile(value_expr)?;

        // Call store-var-by-symbol builtin to update the var at runtime
        let ns_sym_reg = self
            .builder
            .assign_new(IrValue::TaggedConstant((ns_symbol_id << 3) as isize));
        let name_sym_reg = self
            .builder
            .assign_new(IrValue::TaggedConstant((name_symbol_id << 3) as isize));
        self.call_builtin(
            "store-var-by-symbol",
            vec![ns_sym_reg, name_sym_reg, value_reg],
        )?;

        // def returns the value
        Ok(value_reg)
    }

    /// Get current namespace name (for REPL prompt)
    pub fn get_current_namespace(&self) -> String {
        // SAFETY: Read-only access, single-threaded
        unsafe {
            let rt = &*self.runtime.get();
            rt.namespace_name(self.current_namespace_ptr)
        }
    }

    /// Set up user namespace after clojure.core is fully loaded.
    /// This refers all of clojure.core into user and switches to user namespace.
    /// Must be called after loading core.clj.
    pub fn setup_user_namespace(&mut self) {
        unsafe {
            let rt = &mut *self.runtime.get();

            if let (Some(&user_ns_ptr), Some(&core_ns_ptr)) = (
                self.namespace_registry.get("user"),
                self.namespace_registry.get("clojure.core"),
            ) {
                // Refer all of clojure.core into user namespace
                let new_user_ptr = rt.refer_all(user_ns_ptr, core_ns_ptr).unwrap();
                rt.update_namespace_root("user", new_user_ptr);
                self.namespace_registry.insert("user".to_string(), new_user_ptr);
                self.current_namespace_ptr = new_user_ptr;
            }
        }
    }

    /// Sync namespace registry with runtime after GC relocations
    pub fn sync_namespace_registry(&mut self) {
        // SAFETY: Single-threaded access after GC
        unsafe {
            let rt = &*self.runtime.get();

            // Find current namespace name by reverse lookup BEFORE updating registry
            // (Don't dereference current_namespace_ptr - it may be stale after compacting GC)
            let current_ns_name = self
                .namespace_registry
                .iter()
                .find(|(_, ptr)| **ptr == self.current_namespace_ptr)
                .map(|(name, _)| name.clone());

            // Update registry from runtime's namespace roots
            for (name, &ptr) in rt.get_namespace_pointers() {
                if let Some(existing) = self.namespace_registry.get_mut(name) {
                    *existing = ptr;
                }
            }

            // Update current_namespace_ptr using the name we found
            if let Some(name) = current_ns_name
                && let Some(&new_ptr) = rt.get_namespace_pointers().get(&name) {
                    self.current_namespace_ptr = new_ptr;
                }
        }
    }

    /// Compile namespace declaration
    fn compile_ns(&mut self, name: &str) -> Result<IrValue, String> {
        // Check if namespace already exists
        if let Some(&ns_ptr) = self.namespace_registry.get(name) {
            // Switch to existing namespace
            self.current_namespace_ptr = ns_ptr;
        } else {
            // Create new namespace
            // SAFETY: Single-threaded compilation
            let ns_ptr = unsafe {
                let rt = &mut *self.runtime.get();
                let mut ns_ptr = rt.allocate_namespace(name)?;
                rt.add_namespace_root(name.to_string(), ns_ptr);

                // Refer all of clojure.core into the new namespace (like Clojure does)
                // Skip if this IS clojure.core, or if clojure.core doesn't exist yet
                if name != "clojure.core"
                    && let Some(core_ns_ptr) = rt.get_namespace_by_name("clojure.core") {
                        ns_ptr = rt.refer_all(ns_ptr, core_ns_ptr)?;
                        // Update the namespace root since refer_all may have reallocated
                        rt.update_namespace_root(name, ns_ptr);
                    }

                ns_ptr
            };

            self.namespace_registry.insert(name.to_string(), ns_ptr);
            self.current_namespace_ptr = ns_ptr;

            // Track that we used clojure.core (for potential future exclude support)
            self.used_namespaces
                .insert(name.to_string(), vec!["clojure.core".to_string()]);
        }

        // Emit instruction to set *ns* at runtime using symbol-based lookup
        let (core_ns_symbol_id, ns_var_symbol_id) = unsafe {
            let rt = &mut *self.runtime.get();
            let core_ns_id = rt.intern_symbol("clojure.core");
            let ns_var_id = rt.intern_symbol("*ns*");
            (core_ns_id, ns_var_id)
        };

        // Load the current namespace pointer as a tagged constant
        let ns_value = self.builder.new_register();
        self.builder.emit(Instruction::LoadConstant(
            ns_value,
            IrValue::TaggedConstant((self.current_namespace_ptr << 3) as isize),
        ));

        // Store it into *ns* var using symbol-based lookup
        let core_ns_reg = self
            .builder
            .assign_new(IrValue::TaggedConstant((core_ns_symbol_id << 3) as isize));
        let ns_var_reg = self
            .builder
            .assign_new(IrValue::TaggedConstant((ns_var_symbol_id << 3) as isize));
        self.call_builtin(
            "store-var-by-symbol",
            vec![core_ns_reg, ns_var_reg, ns_value],
        )?;

        // Return the namespace pointer (like the value of the ns form)
        Ok(ns_value)
    }

    /// Compile use declaration
    fn compile_use(&mut self, namespace: &str) -> Result<IrValue, String> {
        // Add namespace to used list for current namespace
        // SAFETY: Read-only access, single-threaded
        let current_ns_name = unsafe {
            let rt = &*self.runtime.get();
            rt.namespace_name(self.current_namespace_ptr)
        };

        self.used_namespaces
            .entry(current_ns_name)
            .or_insert_with(|| vec!["clojure.core".to_string()])
            .push(namespace.to_string());

        // Return nil
        let result = self.builder.new_register();
        self.builder
            .emit(Instruction::LoadConstant(result, IrValue::Null));
        Ok(result)
    }

    fn compile_if(
        &mut self,
        test: &Expr,
        then: &Expr,
        else_: &Option<Box<Expr>>,
    ) -> Result<IrValue, String> {
        // Compile test expression
        let test_reg = self.compile(test)?;

        // Create labels for control flow
        let else_label = self.builder.new_label();
        let end_label = self.builder.new_label();

        // Result register
        let result = self.builder.new_register();

        // In Clojure, both false and nil are falsey
        // Load false into a register for comparison
        let false_reg = self.builder.new_register();
        self.builder.emit(Instruction::LoadFalse(false_reg));

        // Jump to else if test is false
        self.builder.emit(Instruction::JumpIf(
            else_label.clone(),
            Condition::Equal,
            test_reg,
            false_reg,
        ));

        // Also jump to else if test is nil
        let nil_reg = self.builder.new_register();
        self.builder
            .emit(Instruction::LoadConstant(nil_reg, IrValue::Null));
        self.builder.emit(Instruction::JumpIf(
            else_label.clone(),
            Condition::Equal,
            test_reg,
            nil_reg,
        ));

        // Then branch
        let then_reg = self.compile(then)?;
        self.builder.emit(Instruction::Assign(result, then_reg));
        self.builder.emit(Instruction::Jump(end_label.clone()));

        // Else branch
        self.builder.emit(Instruction::Label(else_label));
        if let Some(else_expr) = else_ {
            let else_reg = self.compile(else_expr)?;
            self.builder.emit(Instruction::Assign(result, else_reg));
        } else {
            self.builder
                .emit(Instruction::LoadConstant(result, IrValue::Null));
        }

        // End label
        self.builder.emit(Instruction::Label(end_label));

        Ok(result)
    }

    fn compile_do(&mut self, exprs: &[Expr]) -> Result<IrValue, String> {
        let mut last_result = IrValue::Null;

        for expr in exprs {
            last_result = self.compile(expr)?;
        }

        Ok(last_result)
    }

    fn compile_let(
        &mut self,
        bindings: &[(String, Box<Expr>)],
        body: &[Expr],
    ) -> Result<IrValue, String> {
        // (let [x 10 y 20] (+ x y))
        // This creates LEXICAL (stack-allocated) bindings, not dynamic bindings
        //
        // Implementation:
        // 1. Push new scope
        // 2. For each binding:
        //    - Compile value expression (can reference prior bindings!)
        //    - Store result in a register
        //    - Add to current scope
        // 3. Compile body (can reference all bindings)
        // 4. Pop scope
        // 5. Return last body expression's value

        // Push new local scope
        self.push_scope();

        // Compile each binding sequentially
        for (name, value_expr) in bindings {
            // Compile the value expression
            // This can reference prior bindings in this let!
            let value_reg = self.compile(value_expr)?;

            // Bind the name to the register in current scope
            // No need to emit Assign - the value is already in the register
            self.bind_local(name.clone(), value_reg);
        }

        // Compile body (returns last expression)
        let mut result = IrValue::Null;
        for expr in body {
            result = self.compile(expr)?;
        }

        // Pop the local scope
        self.pop_scope();

        Ok(result)
    }

    fn compile_loop(
        &mut self,
        bindings: &[(String, Box<Expr>)],
        body: &[Expr],
    ) -> Result<IrValue, String> {
        self.push_scope();

        // Compile each binding and track registers
        // IMPORTANT: Bindings must be in registers (not TaggedConstants) so recur can assign to them
        let mut binding_registers = Vec::new();
        for (name, value_expr) in bindings {
            let value = self.compile(value_expr)?;
            let value_reg = self.ensure_register(value);
            self.bind_local(name.clone(), value_reg);
            binding_registers.push(value_reg);
        }

        // Create and emit loop label
        let loop_label = self.builder.new_label();
        self.builder.emit(Instruction::Label(loop_label.clone()));

        // Push loop context for recur
        // loop bindings are register-only, no local slots
        let num_bindings = binding_registers.len();
        self.loop_contexts.push(LoopContext {
            label: loop_label,
            binding_registers,
            local_slots: vec![None; num_bindings],
        });

        // Compile body
        let mut result = IrValue::Null;
        for expr in body {
            result = self.compile(expr)?;
        }

        self.loop_contexts.pop();
        self.pop_scope();

        Ok(result)
    }

    fn compile_recur(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        let context = self
            .loop_contexts
            .last()
            .ok_or_else(|| "recur not in loop or fn context".to_string())?
            .clone();

        if args.len() != context.binding_registers.len() {
            return Err(format!(
                "recur arity mismatch: expected {}, got {}",
                context.binding_registers.len(),
                args.len()
            ));
        }

        // 1. Compile ALL new values first (before any assignments)
        let mut new_values = Vec::new();
        for arg in args {
            new_values.push(self.compile(arg)?);
        }

        // 2. Allocate temps and copy new values into them (parallel read)
        // This ensures all source values are captured before any destinations are modified
        let mut temps = Vec::new();
        for new_value in &new_values {
            let temp = self.builder.new_register();
            self.builder.emit(Instruction::Assign(temp, *new_value));
            temps.push(temp);
        }

        // 3. Copy temps into binding registers (parallel write)
        for (binding_reg, temp) in context.binding_registers.iter().zip(temps.iter()) {
            self.builder.emit(Instruction::Assign(*binding_reg, *temp));
        }

        // 3b. Also store to local slots if they exist (for fn params)
        // This is crucial: fn params are bound to local slots, so recur must update them
        for (i, temp) in temps.iter().enumerate() {
            if let Some(Some(local_slot)) = context.local_slots.get(i) {
                self.builder.emit(Instruction::StoreLocal(*local_slot, *temp));
            }
        }

        // 4. Jump to loop label
        self.builder.emit(Instruction::Jump(context.label.clone()));

        // Return dummy (recur never returns normally)
        let dummy = self.builder.new_register();
        self.builder
            .emit(Instruction::LoadConstant(dummy, IrValue::Null));
        Ok(dummy)
    }

    fn compile_binding(
        &mut self,
        bindings: &[(String, Box<Expr>)],
        body: &[Expr],
    ) -> Result<IrValue, String> {
        // (binding [var1 val1 var2 val2 ...] body)
        // 1. Compile and push all bindings
        // 2. Compile body (like do - last expression's value is returned)
        // 3. Pop all bindings in reverse order

        let mut var_ptrs = Vec::new();

        // Compile and push all bindings
        for (var_name, value_expr) in bindings {
            // Parse var_name to get namespace and name
            let (namespace, name) = self.parse_var_name(var_name)?;

            // Look up the var pointer
            let var_ptr = self.lookup_var_pointer(&namespace, &name)?;

            // Compile the value expression
            let value_reg = self.compile(value_expr)?;
            // Ensure value is in a register (not a constant)
            let value_in_reg = self.ensure_register(value_reg);

            // Load var_ptr constant into a register
            let var_ptr_reg = self.builder.new_register();
            self.builder.emit(Instruction::LoadConstant(
                var_ptr_reg,
                IrValue::TaggedConstant(var_ptr as isize),
            ));

            // Emit PushBinding as ExternalCall
            // Call builtin_push_binding(var_ptr, value) -> result
            let builtin_addr = crate::trampoline::builtin_push_binding as usize;
            let push_result = self.builder.new_register();
            self.builder.emit(Instruction::ExternalCall(
                push_result,
                builtin_addr,
                vec![var_ptr_reg, value_in_reg],
            ));

            var_ptrs.push(var_ptr);
        }

        // Compile body (like do - last expression's value is returned)
        let mut result = self.builder.new_register();
        self.builder
            .emit(Instruction::LoadConstant(result, IrValue::Null));

        for expr in body {
            result = self.compile(expr)?;
        }

        // Pop all bindings in reverse order
        for &var_ptr in var_ptrs.iter().rev() {
            // Load var_ptr constant into a register
            let var_ptr_reg = self.builder.new_register();
            self.builder.emit(Instruction::LoadConstant(
                var_ptr_reg,
                IrValue::TaggedConstant(var_ptr as isize),
            ));

            // Emit PopBinding as ExternalCall
            // Call builtin_pop_binding(var_ptr) -> result
            let builtin_addr = crate::trampoline::builtin_pop_binding as usize;
            let pop_result = self.builder.new_register();
            self.builder.emit(Instruction::ExternalCall(
                pop_result,
                builtin_addr,
                vec![var_ptr_reg],
            ));
        }

        Ok(result)
    }

    fn compile_set(&mut self, var_expr: &Expr, value_expr: &Expr) -> Result<IrValue, String> {
        // (set! var value)
        // Parse var name (must be a Var reference)
        let (namespace, name) = match var_expr {
            Expr::Var { namespace, name } => (namespace, name),
            _ => return Err("set! requires a var reference".to_string()),
        };

        // Look up the var pointer
        let var_ptr = self.lookup_var_pointer(namespace, name)?;

        // Compile the value expression
        let value_reg = self.compile(value_expr)?;

        // Emit SetVar as ExternalCall
        // Call builtin_set_binding(var_ptr, value) -> result
        let builtin_addr = crate::trampoline::builtin_set_binding as usize;
        let set_result = self.builder.new_register();
        self.builder.emit(Instruction::ExternalCall(
            set_result,
            builtin_addr,
            vec![IrValue::TaggedConstant(var_ptr as isize), value_reg],
        ));

        // set! returns the new value
        Ok(value_reg)
    }

    /// Parse a var name into namespace and name components
    /// Handles both qualified (ns/name) and unqualified (name) symbols
    fn parse_var_name(&self, var_name: &str) -> Result<(Option<String>, String), String> {
        if let Some(idx) = var_name.find('/') {
            let namespace = var_name[..idx].to_string();
            let name = var_name[idx + 1..].to_string();
            Ok((Some(namespace), name))
        } else {
            Ok((None, var_name.to_string()))
        }
    }

    /// Push a new local scope (for let)
    fn push_scope(&mut self) {
        self.local_scopes.push(HashMap::new());
    }

    /// Pop a local scope (end of let)
    fn pop_scope(&mut self) {
        self.local_scopes.pop();
    }

    /// Bind a local variable to a register in the current scope
    fn bind_local(&mut self, name: String, register: IrValue) {
        if let Some(scope) = self.local_scopes.last_mut() {
            scope.insert(name, Binding::Register(register));
        }
    }

    /// Bind a parameter to a local slot (Beagle pattern)
    fn bind_local_slot(&mut self, name: String, slot: usize) {
        if let Some(scope) = self.local_scopes.last_mut() {
            scope.insert(name, Binding::LocalSlot(slot));
        }
    }

    /// Look up a local variable, searching from innermost to outermost scope
    /// Returns the binding (register or local slot)
    fn lookup_local_binding(&self, name: &str) -> Option<Binding> {
        // Search from innermost scope outward
        for scope in self.local_scopes.iter().rev() {
            if let Some(binding) = scope.get(name) {
                return Some(*binding);
            }
        }
        None
    }

    /// Look up a local variable and get its value as an IrValue
    /// For LocalSlot bindings, emits LoadLocal to load into a fresh register
    fn lookup_local(&mut self, name: &str) -> Option<IrValue> {
        let binding = self.lookup_local_binding(name)?;
        Some(match binding {
            Binding::Register(reg) => reg,
            Binding::LocalSlot(slot) => {
                // Emit LoadLocal to load from stack into a fresh register
                let temp = self.builder.new_register();
                self.builder.emit(Instruction::LoadLocal(temp, slot));
                temp
            }
        })
    }

    /// Look up a var pointer by namespace and name
    fn lookup_var_pointer(&self, namespace: &Option<String>, name: &str) -> Result<usize, String> {
        // SAFETY: Single-threaded REPL - no concurrent access during compilation
        let rt = unsafe { &*self.runtime.get() };

        // Determine which namespace to search
        let ns_ptr = if let Some(ns_name) = namespace {
            *self
                .namespace_registry
                .get(ns_name)
                .ok_or_else(|| format!("Namespace not found: {}", ns_name))?
        } else {
            self.current_namespace_ptr
        };

        // Look up the var
        rt.namespace_lookup(ns_ptr, name)
            .ok_or_else(|| format!("Var not found: {}", name))
    }

    /// Call a builtin function by name
    /// Gets the function pointer directly from the builtin registry and emits a Call instruction.
    /// Returns the result register.
    fn call_builtin(&mut self, builtin_name: &str, args: Vec<IrValue>) -> Result<IrValue, String> {
        // Get the function pointer directly from the builtin registry
        let function_ptr = crate::builtins::get_builtin_descriptors()
            .into_iter()
            .find(|desc| desc.name.ends_with(builtin_name))
            .ok_or_else(|| format!("Unknown builtin: {}", builtin_name))?
            .function_ptr;

        // Tag the function pointer (tag 0b100 for function pointers)
        let tagged_fn_ptr = (function_ptr << 3) | 0b100;

        // Load the function pointer into a register (required by codegen)
        let fn_ptr_reg = self
            .builder
            .assign_new(IrValue::TaggedConstant(tagged_fn_ptr as isize));

        // Ensure all args are in registers (convert constants if needed)
        // This is important: constants in codegen use temp registers which can be
        // clobbered by internal Call computation. By using assign_new here,
        // we ensure proper register allocation.
        let args: Vec<_> = args
            .into_iter()
            .map(|v| self.builder.assign_new(v))
            .collect();

        // Emit a Call instruction with the builtin function pointer
        let result = self.builder.new_register();
        self.builder
            .emit(Instruction::Call(result, fn_ptr_reg, args));
        Ok(result)
    }

    fn compile_fn(
        &mut self,
        name: &Option<String>,
        arities: &[crate::value::FnArity],
    ) -> Result<IrValue, String> {
        // Per-function compilation (Beagle's approach):
        // Each function (or each arity) is compiled separately with its own register allocation

        // Step 1: Analyze closures - find free variables across ALL arities (union)
        let free_vars = self.find_free_variables_across_arities(arities, name)?;

        // DEBUG: Print free vars for protocol methods
        free_vars.is_empty();

        // Step 2: Capture free variables (evaluate them in current scope)
        // IMPORTANT: This must happen BEFORE compiling the function body to capture
        // the correct registers from the outer scope
        let mut closure_values = Vec::new();
        for var_name in &free_vars {
            if let Some(value) = self.lookup_local(var_name) {
                // Ensure closure values are in registers (not TaggedConstant)
                // because MakeFunctionPtr needs physical registers
                let value_reg = self.builder.assign_new(value);
                closure_values.push(value_reg);
            } else {
                return Err(format!("Free variable not found in scope: {}", var_name));
            }
        }

        // Step 3: Compile each arity separately
        let mut compiled_arities: Vec<(usize, usize, bool)> = Vec::new(); // (param_count, code_ptr, is_variadic)

        // Determine if this is a multi-arity function definition
        // Multi-arity functions ALWAYS use the closure calling convention (x0 = fn object)
        // even if they have no captured closures, because the call site doesn't know
        // which arity will be selected.
        let is_multi_arity = arities.len() > 1 || arities.iter().any(|a| a.rest_param.is_some());

        let arity_count = arities.len();
        for arity in arities {
            let is_variadic = arity.rest_param.is_some();
            let param_count = arity.params.len();

            let code_ptr =
                self.compile_single_arity(name, arity, &free_vars, is_multi_arity, arity_count)?;

            compiled_arities.push((param_count, code_ptr, is_variadic));
        }

        // Step 4: Determine variadic minimum and index (if any)
        let variadic_info = compiled_arities
            .iter()
            .enumerate()
            .find(|(_, (_, _, is_var))| *is_var)
            .map(|(idx, (count, _, _))| (*count, idx));
        let variadic_min = variadic_info.map(|(count, _)| count);
        let variadic_index = variadic_info.map(|(_, idx)| idx);

        // Step 5: Create function object
        let fn_obj_reg = self.builder.new_register();
        let closure_count = closure_values.len();

        // For closures: emit PushToStack + CurrentStackPosition sequence (Beagle pattern)
        // This ensures closure values are in FP-relative stack slots that GC can scan
        // Values must be pushed in REVERSE order because:
        // - Stack grows DOWN (lower addresses)
        // - from_raw_parts reads at INCREASING addresses
        // - So values[0] must end up at the LOWEST address (last pushed)
        let values_ptr = if !closure_values.is_empty() {
            // Push each closure value in REVERSE order to FP-relative stack
            for value in closure_values.iter().rev() {
                self.builder.emit(Instruction::PushToStack(*value));
            }

            // After all pushes, get pointer to values[0] (the last pushed value)
            // CurrentStackPosition gives the NEXT available slot, so we add 8
            // to point to where values[0] actually is
            let next_slot_ptr = self.builder.new_register();
            self.builder
                .emit(Instruction::CurrentStackPosition(next_slot_ptr));

            // values_ptr = next_slot_ptr + 8
            let ptr_reg = self.builder.new_register();
            let eight = self.builder.assign_new(IrValue::RawConstant(8));
            self.builder
                .emit(Instruction::AddInt(ptr_reg, next_slot_ptr, eight));

            ptr_reg
        } else {
            // No closures - use a dummy value (won't be used)
            IrValue::TaggedConstant(0)
        };

        if arities.len() == 1 && variadic_min.is_none() {
            // Single fixed arity - create function pointer
            let code_ptr = compiled_arities[0].1;

            self.builder.emit(Instruction::MakeFunctionPtr(
                fn_obj_reg,
                code_ptr,
                values_ptr,
                closure_count,
            ));
        } else {
            // Multi-arity or variadic - use MakeMultiArityFn
            let arity_table: Vec<(usize, usize)> = compiled_arities
                .iter()
                .map(|(param_count, code_ptr, _)| (*param_count, *code_ptr))
                .collect();

            self.builder.emit(Instruction::MakeMultiArityFn(
                fn_obj_reg,
                arity_table,
                variadic_min,
                variadic_index,
                values_ptr,
                closure_count,
            ));
        }

        // Pop closure values from logical stack after allocation
        if closure_count > 0 {
            self.builder.emit(Instruction::PopFromStack(closure_count));
        }

        Ok(fn_obj_reg)
    }

    /// Compile a single arity of a function
    /// `is_multi_arity`: true if this arity is part of a multi-arity definition
    /// `arity_count`: total number of arities (needed for multi-arity closure layout)
    fn compile_single_arity(
        &mut self,
        name: &Option<String>,
        arity: &crate::value::FnArity,
        free_vars: &[String],
        is_multi_arity: bool,
        arity_count: usize,
    ) -> Result<usize, String> {
        // Save the current IR builder and create a new one for this arity
        let outer_builder = std::mem::take(&mut self.builder);

        // Push new scope for function parameters
        self.push_scope();

        // Create a virtual register for the function object (self)
        // For closures AND multi-arity functions, the function object is passed as the first argument (x0)
        // Even if there are no captured closure values, multi-arity functions always receive
        // the function object in x0 because the call site uses the closure calling convention.
        let has_closures = !free_vars.is_empty();

        // For multi-arity functions, we always use closure calling convention (x0 = fn obj)
        // regardless of whether there are actual closure values, because the call site
        // doesn't know which arity will be selected and uses the same calling convention.
        // The self_reg is needed if:
        // 1. There are captured closure values (has_closures), OR
        // 2. This is part of a multi-arity definition (is_multi_arity)
        let uses_closure_convention = has_closures || is_multi_arity;
        let self_reg = if uses_closure_convention {
            Some(IrValue::Register(crate::ir::VirtualRegister::Argument(0)))
        } else {
            None
        };

        // Bind parameters by storing to local slots on stack (Beagle pattern)
        // ARM64 calling convention:
        // - Regular functions (single arity, no closures): arguments are in x0, x1, x2, etc.
        // - Closures/Multi-arity: x0 = closure object, user args in x1, x2, x3, etc.
        //
        // CRITICAL FIX: Store arguments to stack immediately at function entry!
        // The x0-x7 registers are caller-saved and will be clobbered by ANY function call.
        // By storing to local slots (FP-relative stack positions), arguments survive all calls.
        // This follows Beagle's approach in ast.rs lines 665-684.
        let param_offset = if uses_closure_convention { 1 } else { 0 };
        let mut param_registers = Vec::new();
        let mut param_local_slots = Vec::new(); // Track local slots for recur
        for (i, param) in arity.params.iter().enumerate() {
            let arg_reg = IrValue::Register(crate::ir::VirtualRegister::Argument(i + param_offset));
            // Allocate a local slot and store the argument there
            let local_slot = self.builder.allocate_local();
            self.builder
                .emit(Instruction::StoreLocal(local_slot, arg_reg));
            self.bind_local_slot(param.clone(), local_slot);
            // For recur, we still need a register representation
            // Load the parameter into a temp register for the param_registers vec
            let temp_reg = self.builder.new_register();
            self.builder
                .emit(Instruction::LoadLocal(temp_reg, local_slot));
            param_registers.push(temp_reg);
            param_local_slots.push(Some(local_slot)); // Track for recur
        }

        // Bind rest parameter if variadic
        // The rest collection is now passed by the caller as an argument
        // at position fixed_count + param_offset (right after the fixed params)
        if let Some(rest) = &arity.rest_param {
            let fixed_count = arity.params.len();
            let rest_arg_index = fixed_count + param_offset;
            let arg_reg = IrValue::Register(crate::ir::VirtualRegister::Argument(rest_arg_index));
            // Store to local slot like other params
            let local_slot = self.builder.allocate_local();
            self.builder
                .emit(Instruction::StoreLocal(local_slot, arg_reg));
            self.bind_local_slot(rest.clone(), local_slot);
            // For recur, keep track of the register
            let temp_reg = self.builder.new_register();
            self.builder
                .emit(Instruction::LoadLocal(temp_reg, local_slot));
            param_registers.push(temp_reg);
            param_local_slots.push(Some(local_slot)); // Track for recur
        }

        // Bind closure variables
        // Load them from the function object (self_reg = x0 for closures)
        // For multi-arity functions, closure values are at a different offset (after arity table)
        if let Some(self_reg) = &self_reg {
            for (i, var_name) in free_vars.iter().enumerate() {
                let closure_reg = self.builder.new_register();
                if is_multi_arity {
                    // Multi-arity: closures are stored after the arity table
                    self.builder.emit(Instruction::LoadClosureMultiArity(
                        closure_reg,
                        *self_reg,
                        arity_count,
                        i,
                    ));
                } else {
                    // Single-arity: use standard closure layout
                    self.builder
                        .emit(Instruction::LoadClosure(closure_reg, *self_reg, i));
                }
                self.bind_local(var_name.clone(), closure_reg);
            }
        }

        // Compile and check pre-conditions
        // Each :pre condition must be truthy, otherwise throw AssertionError
        for (i, pre_cond) in arity.pre_conditions.iter().enumerate() {
            let cond_result = self.compile(pre_cond)?;
            self.builder.emit(Instruction::AssertPre(cond_result, i));
        }

        // Push loop context for recur in fn body
        let fn_entry_label = self.builder.new_label();
        self.builder
            .emit(Instruction::Label(fn_entry_label.clone()));
        self.loop_contexts.push(LoopContext {
            label: fn_entry_label,
            binding_registers: param_registers,
            local_slots: param_local_slots,
        });

        // Compile body (implicit do)
        let result = self.compile_body(&arity.body)?;

        // Pop fn's loop context
        self.loop_contexts.pop();

        // Compile and check post-conditions
        // Each :post condition can reference % which is bound to the result
        if !arity.post_conditions.is_empty() {
            // Bind % to the result for post-condition expressions
            self.bind_local("%".to_string(), result);

            for (i, post_cond) in arity.post_conditions.iter().enumerate() {
                let cond_result = self.compile(post_cond)?;
                self.builder.emit(Instruction::AssertPost(cond_result, i));
            }
        }

        // Return result
        self.builder.emit(Instruction::Ret(result));

        // Pop function scope
        self.pop_scope();

        // Compile this arity's IR to native code
        let reserved_exception_slots = self.builder.reserved_exception_slots;
        let fn_instructions = self.builder.take_instructions();

        // Store the function IR for display purposes
        self.compiled_function_irs
            .push((name.clone(), fn_instructions.clone()));

        // Pass num_locals from the builder (includes all allocated local slots)
        let num_locals = self.builder.num_locals;
        let compiled =
            Arm64CodeGen::compile_function(&fn_instructions, num_locals, reserved_exception_slots)?;

        // Register stack map with runtime for GC
        // NOTE: number_of_locals must be the TOTAL stack slots (spills + exceptions + locals)
        // so that GC can find dynamic saves pushed after the reserved slots
        unsafe {
            let rt = &mut *self.runtime.get();
            for (pc, stack_size) in &compiled.stack_map {
                rt.add_stack_map_entry(*pc, crate::gc::StackMapDetails {
                    function_name: None,
                    number_of_locals: compiled.num_stack_slots,
                    current_stack_size: *stack_size,
                    max_stack_size: *stack_size,
                });
            }
        }

        // Restore the outer IR builder
        self.builder = outer_builder;

        Ok(compiled.code_ptr)
    }

    /// Find free variables across all arities (union)
    fn find_free_variables_across_arities(
        &self,
        arities: &[crate::value::FnArity],
        fn_name: &Option<String>,
    ) -> Result<Vec<String>, String> {
        let mut all_free_vars = std::collections::HashSet::new();

        for arity in arities {
            let arity_free_vars = self.find_free_variables_in_arity(arity, fn_name)?;
            all_free_vars.extend(arity_free_vars);
        }

        // Return as sorted vector for consistent ordering
        let mut result: Vec<String> = all_free_vars.into_iter().collect();
        result.sort();
        Ok(result)
    }

    /// Compile a sequence of expressions (like do)
    fn compile_body(&mut self, exprs: &[Expr]) -> Result<IrValue, String> {
        if exprs.is_empty() {
            let result = self.builder.new_register();
            self.builder
                .emit(Instruction::LoadConstant(result, IrValue::Null));
            return Ok(result);
        }

        let mut result = self.builder.new_register();
        for expr in exprs {
            result = self.compile(expr)?;
        }

        Ok(result)
    }

    /// Find free variables in a function arity
    /// A variable is free if:
    /// 1. It's referenced in the body
    /// 2. It's not a parameter
    /// 3. It's not bound by an inner let
    /// 4. It's a local (not a global var or builtin)
    fn find_free_variables_in_arity(
        &self,
        arity: &crate::value::FnArity,
        fn_name: &Option<String>,
    ) -> Result<Vec<String>, String> {
        let mut bound_vars = std::collections::HashSet::new();

        // Add parameters to bound vars
        for param in &arity.params {
            bound_vars.insert(param.clone());
        }

        // Add rest param if present
        if let Some(rest) = &arity.rest_param {
            bound_vars.insert(rest.clone());
        }

        // Add function name if present (for self-recursion)
        if let Some(name) = fn_name {
            bound_vars.insert(name.clone());
        }

        let mut free_vars = std::collections::HashSet::new();

        // Analyze pre-condition expressions
        for expr in &arity.pre_conditions {
            self.collect_free_vars_from_expr(expr, &bound_vars, &mut free_vars);
        }

        // Analyze post-condition expressions
        // Note: % is a special binding for the return value, but it's not a free variable
        for expr in &arity.post_conditions {
            self.collect_free_vars_from_expr(expr, &bound_vars, &mut free_vars);
        }

        // Analyze body expressions
        for expr in &arity.body {
            self.collect_free_vars_from_expr(expr, &bound_vars, &mut free_vars);
        }

        // Return as sorted vector for consistent ordering
        let mut result: Vec<String> = free_vars.into_iter().collect();
        result.sort();
        Ok(result)
    }

    /// Recursively collect free variables from an expression
    fn collect_free_vars_from_expr(
        &self,
        expr: &Expr,
        bound: &std::collections::HashSet<String>,
        free: &mut std::collections::HashSet<String>,
    ) {
        match expr {
            Expr::Var {
                namespace: None,
                name,
            } => {
                // Only consider unqualified names
                if !bound.contains(name) && !self.is_builtin(name) {
                    // Check if it's a local variable in current scope
                    if self.lookup_local_binding(name).is_some() {
                        // It's a free variable (captured from outer scope)
                        free.insert(name.clone());
                    }
                    // If not in local scope, it's a global var, not free
                }
            }

            Expr::Let { bindings, body } => {
                // Let creates new bindings
                let mut new_bound = bound.clone();

                for (name, value_expr) in bindings {
                    // Analyze value expression with current bindings
                    self.collect_free_vars_from_expr(value_expr, &new_bound, free);
                    // Add this binding for subsequent expressions
                    new_bound.insert(name.clone());
                }

                // Analyze body with all bindings
                for expr in body {
                    self.collect_free_vars_from_expr(expr, &new_bound, free);
                }
            }

            Expr::Fn {
                name: fn_name,
                arities,
            } => {
                // Nested function - need to recurse with proper scoping
                let mut fn_bound = bound.clone();

                // Name (if present) is bound within the function
                if let Some(n) = fn_name {
                    fn_bound.insert(n.clone());
                }

                for arity in arities {
                    let mut arity_bound = fn_bound.clone();

                    // Add parameters
                    for param in &arity.params {
                        arity_bound.insert(param.clone());
                    }

                    // Add rest param
                    if let Some(rest) = &arity.rest_param {
                        arity_bound.insert(rest.clone());
                    }

                    // Analyze body
                    for expr in &arity.body {
                        self.collect_free_vars_from_expr(expr, &arity_bound, free);
                    }
                }
            }

            Expr::If { test, then, else_ } => {
                self.collect_free_vars_from_expr(test, bound, free);
                self.collect_free_vars_from_expr(then, bound, free);
                if let Some(e) = else_ {
                    self.collect_free_vars_from_expr(e, bound, free);
                }
            }

            Expr::Do { exprs } => {
                for expr in exprs {
                    self.collect_free_vars_from_expr(expr, bound, free);
                }
            }

            Expr::Call { func, args } => {
                self.collect_free_vars_from_expr(func, bound, free);
                for arg in args {
                    self.collect_free_vars_from_expr(arg, bound, free);
                }
            }

            Expr::Def { value, .. } => {
                self.collect_free_vars_from_expr(value, bound, free);
            }

            Expr::Set { var, value } => {
                self.collect_free_vars_from_expr(var, bound, free);
                self.collect_free_vars_from_expr(value, bound, free);
            }

            Expr::Binding { bindings, body } => {
                for (_, value_expr) in bindings {
                    self.collect_free_vars_from_expr(value_expr, bound, free);
                }
                for expr in body {
                    self.collect_free_vars_from_expr(expr, bound, free);
                }
            }

            Expr::Loop { bindings, body } => {
                let mut new_bound = bound.clone();
                for (name, value_expr) in bindings {
                    self.collect_free_vars_from_expr(value_expr, &new_bound, free);
                    new_bound.insert(name.clone());
                }
                for expr in body {
                    self.collect_free_vars_from_expr(expr, &new_bound, free);
                }
            }

            Expr::Recur { args } => {
                for arg in args {
                    self.collect_free_vars_from_expr(arg, bound, free);
                }
            }

            // These don't contain free variables
            Expr::Literal(_) | Expr::Quote(_) | Expr::Ns { .. } | Expr::Use { .. } => {}

            // Qualified vars are not free (they're global)
            Expr::Var {
                namespace: Some(_), ..
            } => {}

            // VarRef always refers to a global var, not free
            Expr::VarRef { .. } => {}

            // DefType is a declaration, no free vars
            Expr::DefType { .. } => {}

            // TypeConstruct: check args for free vars
            Expr::TypeConstruct { args, .. } => {
                for arg in args {
                    self.collect_free_vars_from_expr(arg, bound, free);
                }
            }

            // FieldAccess: check object for free vars
            Expr::FieldAccess { object, .. } => {
                self.collect_free_vars_from_expr(object, bound, free);
            }

            // FieldSet: check object and value for free vars
            Expr::FieldSet { object, value, .. } => {
                self.collect_free_vars_from_expr(object, bound, free);
                self.collect_free_vars_from_expr(value, bound, free);
            }

            // MethodCall: check object and args for free vars
            Expr::MethodCall { object, args, .. } => {
                self.collect_free_vars_from_expr(object, bound, free);
                for arg in args {
                    self.collect_free_vars_from_expr(arg, bound, free);
                }
            }

            // Throw: check exception expression for free vars
            Expr::Throw { exception } => {
                self.collect_free_vars_from_expr(exception, bound, free);
            }

            // Try: check body, catches, and finally for free vars
            Expr::Try {
                body,
                catches,
                finally,
            } => {
                for expr in body {
                    self.collect_free_vars_from_expr(expr, bound, free);
                }
                for catch in catches {
                    // The catch binding is bound within the catch body
                    let mut catch_bound = bound.clone();
                    catch_bound.insert(catch.binding.clone());
                    for expr in &catch.body {
                        self.collect_free_vars_from_expr(expr, &catch_bound, free);
                    }
                }
                if let Some(finally_body) = finally {
                    for expr in finally_body {
                        self.collect_free_vars_from_expr(expr, bound, free);
                    }
                }
            }

            // Protocol-related expressions
            Expr::DefProtocol { .. } => {
                // DefProtocol is a declaration, no free vars
            }

            Expr::ExtendType {
                implementations, ..
            } => {
                // Check method bodies for free vars
                for impl_ in implementations {
                    for method in &impl_.methods {
                        // Parameters are bound within the method body
                        let mut method_bound = bound.clone();
                        for param in &method.params {
                            method_bound.insert(param.clone());
                        }
                        for expr in &method.body {
                            self.collect_free_vars_from_expr(expr, &method_bound, free);
                        }
                    }
                }
            }

            Expr::ProtocolCall { args, .. } => {
                // Check args for free vars
                for arg in args {
                    self.collect_free_vars_from_expr(arg, bound, free);
                }
            }

            Expr::Debugger { expr } => {
                self.collect_free_vars_from_expr(expr, bound, free);
            }

            // TopLevelDo is only used at the top level and handled specially
            // It should never appear in a context where we're collecting free vars
            Expr::TopLevelDo { .. } => {}
        }
    }

    fn compile_call(&mut self, func: &Expr, args: &[Expr]) -> Result<IrValue, String> {
        // Check if it's a built-in first
        if let Expr::Var { namespace, name } = func {
            // Check if it's a built-in (either explicitly qualified or unqualified)
            // IMPORTANT: For unqualified names, we must check that there's no local
            // shadowing the builtin. E.g., (let [+ my-fn] (+ 1 2)) should NOT inline.
            let is_builtin = match namespace {
                Some(ns) if ns == "clojure.core" => self.is_builtin(name),
                None => self.is_builtin(name) && self.lookup_local_binding(name).is_none(),
                _ => false,
            };

            if is_builtin {
                // Dispatch to appropriate built-in handler
                return match name.as_str() {
                    "prim-add" => self.compile_builtin_add(args),
                    "prim-sub" => self.compile_builtin_sub(args),
                    "prim-mul" => self.compile_builtin_mul(args),
                    "prim-div" => self.compile_builtin_div(args),
                    "<" => self.compile_builtin_lt(args),
                    ">" => self.compile_builtin_gt(args),
                    "<=" => self.compile_builtin_le(args),
                    ">=" => self.compile_builtin_ge(args),
                    "=" => self.compile_builtin_eq(args),
                    "__gc" => self.compile_builtin_gc(args),
                    "bit-and" => self.compile_builtin_bit_and(args),
                    "bit-or" => self.compile_builtin_bit_or(args),
                    "bit-xor" => self.compile_builtin_bit_xor(args),
                    "bit-not" => self.compile_builtin_bit_not(args),
                    "bit-shift-left" => self.compile_builtin_bit_shift_left(args),
                    "bit-shift-right" => self.compile_builtin_bit_shift_right(args),
                    "unsigned-bit-shift-right" => {
                        self.compile_builtin_unsigned_bit_shift_right(args)
                    }
                    "bit-shift-right-zero-fill" => {
                        self.compile_builtin_unsigned_bit_shift_right(args)
                    } // CLJS alias
                    "nil?" => self.compile_builtin_nil_pred(args),
                    "number?" => self.compile_builtin_number_pred(args),
                    "string?" => self.compile_builtin_string_pred(args),
                    "fn?" => self.compile_builtin_fn_pred(args),
                    "identical?" => self.compile_builtin_identical(args),
                    "instance?" => self.compile_builtin_instance_check(args),
                    "satisfies?" => self.compile_builtin_satisfies_check(args),
                    "keyword?" => self.compile_builtin_keyword_check(args),
                    "hash-primitive" => self.compile_builtin_hash_primitive(args),
                    "cons?" => self.compile_builtin_cons_check(args),
                    "cons-first" => self.compile_builtin_cons_first(args),
                    "cons-rest" => self.compile_builtin_cons_rest(args),
                    "_println" => self.compile_builtin_println_value(args),
                    "_print" => self.compile_builtin_print_value(args),
                    "_newline" => self.compile_builtin_newline(args),
                    "_print-space" => self.compile_builtin_print_space(args),
                    "__make_reader_symbol_1" => self.compile_builtin_make_reader_symbol_1(args),
                    "__make_reader_symbol_2" => self.compile_builtin_make_reader_symbol_2(args),
                    "__make_reader_list_0" => self.compile_builtin_make_reader_list_0(args),
                    "__make_reader_list_from_vec" => self.compile_builtin_make_reader_list_from_vec(args),
                    "__make_reader_vector_0" => self.compile_builtin_make_reader_vector_0(args),
                    "__make_reader_vector_from_list" => self.compile_builtin_make_reader_vector_from_list(args),
                    "__make_reader_map_0" => self.compile_builtin_make_reader_map_0(args),
                    "__reader_cons" => self.compile_builtin_reader_cons(args),
                    "__reader_list_1" => self.compile_builtin_reader_list_n(args, 1),
                    "__reader_list_2" => self.compile_builtin_reader_list_n(args, 2),
                    "__reader_list_3" => self.compile_builtin_reader_list_n(args, 3),
                    "__reader_list_4" => self.compile_builtin_reader_list_n(args, 4),
                    "__reader_list_5" => self.compile_builtin_reader_list_n(args, 5),
                    // Reader primitive operations for protocol implementations
                    "__reader_list_first" => self.compile_builtin_reader_prim_1(args, "__reader_list_first"),
                    "__reader_list_rest" => self.compile_builtin_reader_prim_1(args, "__reader_list_rest"),
                    "__reader_list_count" => self.compile_builtin_reader_prim_1(args, "__reader_list_count"),
                    "__reader_list_nth" => self.compile_builtin_reader_prim_2(args, "__reader_list_nth"),
                    "__reader_list_conj" => self.compile_builtin_reader_prim_2(args, "__reader_list_conj"),
                    "__reader_vector_first" => self.compile_builtin_reader_prim_1(args, "__reader_vector_first"),
                    "__reader_vector_rest" => self.compile_builtin_reader_prim_1(args, "__reader_vector_rest"),
                    "__reader_vector_count" => self.compile_builtin_reader_prim_1(args, "__reader_vector_count"),
                    "__reader_vector_nth" => self.compile_builtin_reader_prim_2(args, "__reader_vector_nth"),
                    "__reader_vector_conj" => self.compile_builtin_reader_prim_2(args, "__reader_vector_conj"),
                    // Reader type predicates
                    "__reader_list?" => self.compile_builtin_reader_prim_1(args, "__reader_list?"),
                    "__reader_vector?" => self.compile_builtin_reader_prim_1(args, "__reader_vector?"),
                    "__reader_map?" => self.compile_builtin_reader_prim_1(args, "__reader_map?"),
                    "__reader_symbol?" => self.compile_builtin_reader_prim_1(args, "__reader_symbol?"),
                    // New builtins for defmacro support
                    "__is_string" => self.compile_builtin_reader_prim_1(args, "__is_string"),
                    "__is_symbol" => self.compile_builtin_reader_prim_1(args, "__is_symbol"),
                    "__set_macro!" => self.compile_builtin_reader_prim_1(args, "__set_macro!"),
                    "__symbol_1" => self.compile_builtin_reader_prim_1(args, "__symbol_1"),
                    "__symbol_2" => self.compile_builtin_reader_prim_2(args, "__symbol_2"),
                    "__gensym_0" => self.compile_builtin_gensym_0(args),
                    "__gensym_1" => self.compile_builtin_reader_prim_1(args, "__gensym_1"),
                    // Array operations
                    "make-array" => self.compile_builtin_make_array(args),
                    "aget" => self.compile_builtin_aget(args),
                    "aset" => self.compile_builtin_aset(args),
                    "aset!" => self.compile_builtin_aset(args),
                    "alength" => self.compile_builtin_alength(args),
                    "aclone" => self.compile_builtin_aclone(args),
                    // Core function builtins
                    "__keyword_name" => self.compile_builtin_keyword_name(args),
                    "__apply" => self.compile_builtin_apply(args),
                    "__str_concat" => self.compile_builtin_str_concat(args),
                    // Reader set operations
                    "__reader_set_count" => self.compile_builtin_reader_prim_1(args, "__reader_set_count"),
                    "__reader_set_contains" => self.compile_builtin_reader_prim_2(args, "__reader_set_contains"),
                    "__reader_set_conj" => self.compile_builtin_reader_set_conj(args),
                    "__reader_set_get" => self.compile_builtin_reader_prim_2(args, "__reader_set_get"),
                    "__is_reader_set" => self.compile_builtin_reader_prim_1(args, "__is_reader_set"),
                    _ => unreachable!(),
                };
            }
        }

        // General function call (user-defined or first-class function)
        // 1. Evaluate function expression to get function object
        let fn_val = self.compile(func)?;

        // 2. Evaluate arguments
        let mut arg_vals = Vec::new();
        for arg in args {
            arg_vals.push(self.compile(arg)?);
        }

        // 3. Ensure all values are in registers (convert constants if needed)
        // This is important: constants in codegen use temp registers which can be
        // clobbered by internal Call computation. By using assign_new here,
        // we ensure proper register allocation.
        let fn_val = self.builder.assign_new(fn_val);
        let arg_vals: Vec<_> = arg_vals
            .into_iter()
            .map(|v| self.builder.assign_new(v))
            .collect();

        // 4. Emit Call instruction
        let result = self.builder.new_register();
        self.builder
            .emit(Instruction::Call(result, fn_val, arg_vals));

        Ok(result)
    }

    /// Helper to compile polymorphic arithmetic (supports both int and float)
    /// int_op: the integer IR instruction (e.g., AddInt)
    /// float_op: the float IR instruction (e.g., AddFloat)
    fn compile_polymorphic_arith(
        &mut self,
        left: IrValue,
        right: IrValue,
        int_op: fn(IrValue, IrValue, IrValue) -> Instruction,
        float_op: fn(IrValue, IrValue, IrValue) -> Instruction,
    ) -> Result<IrValue, String> {
        // Convert operands to registers at point of use (Beagle pattern)
        // This is called AFTER inner expressions are compiled, so constants
        // get registers that live only during this operation, not across nested calls.
        let left = self.builder.assign_new(left);
        let right = self.builder.assign_new(right);

        // Result register that both paths will write to
        let result = self.builder.new_register();

        // Labels for branching
        let float_path = self.builder.new_label();
        let int_int_path = self.builder.new_label();
        let done = self.builder.new_label();

        // Check if left is int (tag == 0b000)
        let left_tag = self.builder.new_register();
        self.builder.emit(Instruction::GetTag(left_tag, left));
        // If left_tag != 0, it's not an int -> go to float path
        self.builder.emit(Instruction::JumpIf(
            float_path.clone(),
            Condition::NotEqual,
            left_tag,
            IrValue::TaggedConstant(0),
        ));

        // Left is int, check if right is also int
        let right_tag = self.builder.new_register();
        self.builder.emit(Instruction::GetTag(right_tag, right));
        self.builder.emit(Instruction::JumpIf(
            float_path.clone(),
            Condition::NotEqual,
            right_tag,
            IrValue::TaggedConstant(0),
        ));

        // INT + INT path
        self.builder.emit(Instruction::Label(int_int_path));
        let left_untagged = self.builder.new_register();
        let right_untagged = self.builder.new_register();
        self.builder.emit(Instruction::Untag(left_untagged, left));
        self.builder.emit(Instruction::Untag(right_untagged, right));

        let int_result = self.builder.new_register();
        self.builder
            .emit(int_op(int_result, left_untagged, right_untagged));

        // Tag as int (tag 0b000)
        self.builder.emit(Instruction::Tag(
            result,
            int_result,
            IrValue::TaggedConstant(0),
        ));
        self.builder.emit(Instruction::Jump(done.clone()));

        // FLOAT path (at least one operand is non-int)
        // Must verify it's actually a float (tag 0b001), otherwise throw type error
        self.builder.emit(Instruction::Label(float_path));

        // Convert left to float if needed
        let left_float = self.builder.new_register();
        let left_tag2 = self.builder.new_register();
        self.builder.emit(Instruction::GetTag(left_tag2, left));
        let left_is_int = self.builder.new_label();
        let left_is_float = self.builder.new_label();
        let left_convert_done = self.builder.new_label();

        // If left_tag == 0 (int), convert to float
        self.builder.emit(Instruction::JumpIf(
            left_is_int.clone(),
            Condition::Equal,
            left_tag2,
            IrValue::TaggedConstant(0),
        ));

        // Not int - check if it's float (tag == 1)
        self.builder.emit(Instruction::JumpIf(
            left_is_float.clone(),
            Condition::Equal,
            left_tag2,
            IrValue::TaggedConstant(1),
        ));

        // Not int or float - throw type error
        // For now, treat as 0.0 (we need proper exception handling for a real error)
        // TODO: Implement proper type error throwing
        self.builder.emit(Instruction::LoadConstant(
            left_float,
            IrValue::TaggedConstant(0),
        ));
        self.builder
            .emit(Instruction::IntToFloat(left_float, left_float));
        self.builder
            .emit(Instruction::Jump(left_convert_done.clone()));

        // Left is int, convert to float
        self.builder.emit(Instruction::Label(left_is_int));
        let left_int_untagged = self.builder.new_register();
        self.builder
            .emit(Instruction::Untag(left_int_untagged, left));
        self.builder
            .emit(Instruction::IntToFloat(left_float, left_int_untagged));
        self.builder
            .emit(Instruction::Jump(left_convert_done.clone()));

        // Left is already float, load f64 bits from heap
        self.builder.emit(Instruction::Label(left_is_float));
        self.builder.emit(Instruction::LoadFloat(left_float, left));

        self.builder.emit(Instruction::Label(left_convert_done));

        // Convert right to float if needed
        let right_float = self.builder.new_register();
        let right_tag2 = self.builder.new_register();
        self.builder.emit(Instruction::GetTag(right_tag2, right));
        let right_is_int = self.builder.new_label();
        let right_is_float = self.builder.new_label();
        let right_convert_done = self.builder.new_label();

        // If right_tag == 0 (int), convert to float
        self.builder.emit(Instruction::JumpIf(
            right_is_int.clone(),
            Condition::Equal,
            right_tag2,
            IrValue::TaggedConstant(0),
        ));

        // Not int - check if it's float (tag == 1)
        self.builder.emit(Instruction::JumpIf(
            right_is_float.clone(),
            Condition::Equal,
            right_tag2,
            IrValue::TaggedConstant(1),
        ));

        // Not int or float - throw type error
        // For now, treat as 0.0 (we need proper exception handling for a real error)
        // TODO: Implement proper type error throwing
        self.builder.emit(Instruction::LoadConstant(
            right_float,
            IrValue::TaggedConstant(0),
        ));
        self.builder
            .emit(Instruction::IntToFloat(right_float, right_float));
        self.builder
            .emit(Instruction::Jump(right_convert_done.clone()));

        // Right is int, convert to float
        self.builder.emit(Instruction::Label(right_is_int));
        let right_int_untagged = self.builder.new_register();
        self.builder
            .emit(Instruction::Untag(right_int_untagged, right));
        self.builder
            .emit(Instruction::IntToFloat(right_float, right_int_untagged));
        self.builder
            .emit(Instruction::Jump(right_convert_done.clone()));

        // Right is already float, load f64 bits from heap
        self.builder.emit(Instruction::Label(right_is_float));
        self.builder
            .emit(Instruction::LoadFloat(right_float, right));

        self.builder.emit(Instruction::Label(right_convert_done));

        // Perform float operation (on raw f64 bits)
        let float_result = self.builder.new_register();
        self.builder
            .emit(float_op(float_result, left_float, right_float));

        // Allocate new float on heap and get tagged pointer
        // Call builtin_allocate_float(frame_pointer, f64_bits) -> tagged_ptr
        let builtin_addr = crate::trampoline::builtin_allocate_float as usize;
        // Use FramePointer (x29) for GC stack walking. gc_return_addr is computed internally.
        self.builder.emit(Instruction::ExternalCall(
            result,
            builtin_addr,
            vec![IrValue::FramePointer, float_result],
        ));

        self.builder.emit(Instruction::Label(done));

        Ok(result)
    }

    fn compile_builtin_add(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        if args.len() != 2 {
            return Err(format!("+ requires 2 arguments, got {}", args.len()));
        }

        let left = self.compile(&args[0])?;
        let right = self.compile(&args[1])?;

        self.compile_polymorphic_arith(left, right, Instruction::AddInt, Instruction::AddFloat)
    }

    fn compile_builtin_sub(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        if args.len() != 2 {
            return Err(format!("- requires 2 arguments, got {}", args.len()));
        }

        let left = self.compile(&args[0])?;
        let right = self.compile(&args[1])?;

        self.compile_polymorphic_arith(left, right, Instruction::Sub, Instruction::SubFloat)
    }

    fn compile_builtin_mul(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        if args.len() != 2 {
            return Err(format!("* requires 2 arguments, got {}", args.len()));
        }

        let left = self.compile(&args[0])?;
        let right = self.compile(&args[1])?;

        self.compile_polymorphic_arith(left, right, Instruction::Mul, Instruction::MulFloat)
    }

    fn compile_builtin_div(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        if args.len() != 2 {
            return Err(format!("/ requires 2 arguments, got {}", args.len()));
        }

        let left = self.compile(&args[0])?;
        let right = self.compile(&args[1])?;

        self.compile_polymorphic_arith(left, right, Instruction::Div, Instruction::DivFloat)
    }

    fn compile_builtin_lt(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        if args.len() != 2 {
            return Err(format!("< requires 2 arguments, got {}", args.len()));
        }

        let left = self.compile(&args[0])?;
        let right = self.compile(&args[1])?;

        // Ensure operands are in registers (convert constants if needed)
        let left = self.builder.assign_new(left);
        let right = self.builder.assign_new(right);

        let left_untagged = self.builder.new_register();
        let right_untagged = self.builder.new_register();
        self.builder.emit(Instruction::Untag(left_untagged, left));
        self.builder.emit(Instruction::Untag(right_untagged, right));

        let result = self.builder.new_register();
        self.builder.emit(Instruction::Compare(
            result,
            left_untagged,
            right_untagged,
            Condition::LessThan,
        ));

        // Compare now returns properly tagged boolean (3 or 11)
        Ok(result)
    }

    fn compile_builtin_gt(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        if args.len() != 2 {
            return Err(format!("> requires 2 arguments, got {}", args.len()));
        }

        let left = self.compile(&args[0])?;
        let right = self.compile(&args[1])?;

        // Ensure operands are in registers (convert constants if needed)
        let left = self.builder.assign_new(left);
        let right = self.builder.assign_new(right);

        let left_untagged = self.builder.new_register();
        let right_untagged = self.builder.new_register();
        self.builder.emit(Instruction::Untag(left_untagged, left));
        self.builder.emit(Instruction::Untag(right_untagged, right));

        let result = self.builder.new_register();
        self.builder.emit(Instruction::Compare(
            result,
            left_untagged,
            right_untagged,
            Condition::GreaterThan,
        ));

        // Compare now returns properly tagged boolean (3 or 11)
        Ok(result)
    }

    fn compile_builtin_le(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        if args.len() != 2 {
            return Err(format!("<= requires 2 arguments, got {}", args.len()));
        }

        let left = self.compile(&args[0])?;
        let right = self.compile(&args[1])?;

        // Ensure operands are in registers (convert constants if needed)
        let left = self.builder.assign_new(left);
        let right = self.builder.assign_new(right);

        let left_untagged = self.builder.new_register();
        let right_untagged = self.builder.new_register();
        self.builder.emit(Instruction::Untag(left_untagged, left));
        self.builder.emit(Instruction::Untag(right_untagged, right));

        let result = self.builder.new_register();
        self.builder.emit(Instruction::Compare(
            result,
            left_untagged,
            right_untagged,
            Condition::LessThanOrEqual,
        ));

        Ok(result)
    }

    fn compile_builtin_ge(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        if args.len() != 2 {
            return Err(format!(">= requires 2 arguments, got {}", args.len()));
        }

        let left = self.compile(&args[0])?;
        let right = self.compile(&args[1])?;

        // Ensure operands are in registers (convert constants if needed)
        let left = self.builder.assign_new(left);
        let right = self.builder.assign_new(right);

        let left_untagged = self.builder.new_register();
        let right_untagged = self.builder.new_register();
        self.builder.emit(Instruction::Untag(left_untagged, left));
        self.builder.emit(Instruction::Untag(right_untagged, right));

        let result = self.builder.new_register();
        self.builder.emit(Instruction::Compare(
            result,
            left_untagged,
            right_untagged,
            Condition::GreaterThanOrEqual,
        ));

        Ok(result)
    }

    /// _println - prints a single value with newline (like Beagle's pattern)
    fn compile_builtin_println_value(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        if args.len() != 1 {
            return Err(format!("_println requires 1 argument, got {}", args.len()));
        }
        let val = self.compile(&args[0])?;
        let val_reg = self.builder.assign_new(val);
        self.call_builtin("runtime.builtin/_println", vec![val_reg])
    }

    /// _print - prints a single value without newline (like Beagle's pattern)
    fn compile_builtin_print_value(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        if args.len() != 1 {
            return Err(format!("_print requires 1 argument, got {}", args.len()));
        }
        let val = self.compile(&args[0])?;
        let val_reg = self.builder.assign_new(val);
        self.call_builtin("runtime.builtin/_print", vec![val_reg])
    }

    /// _newline - prints just a newline
    fn compile_builtin_newline(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        if !args.is_empty() {
            return Err(format!("_newline takes no arguments, got {}", args.len()));
        }
        self.call_builtin("runtime.builtin/_newline", vec![])
    }

    /// _print-space - prints just a space
    fn compile_builtin_print_space(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        if !args.is_empty() {
            return Err(format!(
                "_print-space takes no arguments, got {}",
                args.len()
            ));
        }
        self.call_builtin("runtime.builtin/_print-space", vec![])
    }

    // __make_reader_symbol_1 - create a ReaderSymbol with just a name
    fn compile_builtin_make_reader_symbol_1(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        if args.len() != 1 {
            return Err(format!("__make_reader_symbol_1 requires 1 argument, got {}", args.len()));
        }
        let name = self.compile(&args[0])?;
        let name_reg = self.builder.assign_new(name);
        self.call_builtin("__make_reader_symbol_1", vec![name_reg])
    }

    // __make_reader_symbol_2 - create a ReaderSymbol with namespace and name
    fn compile_builtin_make_reader_symbol_2(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        if args.len() != 2 {
            return Err(format!("__make_reader_symbol_2 requires 2 arguments, got {}", args.len()));
        }
        let ns = self.compile(&args[0])?;
        let name = self.compile(&args[1])?;
        let ns_reg = self.builder.assign_new(ns);
        let name_reg = self.builder.assign_new(name);
        self.call_builtin("__make_reader_symbol_2", vec![ns_reg, name_reg])
    }

    // __make_reader_list_0 - create an empty ReaderList
    fn compile_builtin_make_reader_list_0(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        if !args.is_empty() {
            return Err(format!("__make_reader_list_0 takes no arguments, got {}", args.len()));
        }
        self.call_builtin("__make_reader_list_0", vec![])
    }

    // __make_reader_list_from_vec - create a ReaderList from a vector
    fn compile_builtin_make_reader_list_from_vec(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        if args.len() != 1 {
            return Err(format!("__make_reader_list_from_vec requires 1 argument, got {}", args.len()));
        }
        let vec = self.compile(&args[0])?;
        let vec_reg = self.builder.assign_new(vec);
        self.call_builtin("__make_reader_list_from_vec", vec![vec_reg])
    }

    // __make_reader_vector_0 - create an empty ReaderVector
    fn compile_builtin_make_reader_vector_0(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        if !args.is_empty() {
            return Err(format!("__make_reader_vector_0 takes no arguments, got {}", args.len()));
        }
        self.call_builtin("__make_reader_vector_0", vec![])
    }

    // __make_reader_vector_from_list - create a ReaderVector from a list
    fn compile_builtin_make_reader_vector_from_list(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        if args.len() != 1 {
            return Err(format!("__make_reader_vector_from_list requires 1 argument, got {}", args.len()));
        }
        let list = self.compile(&args[0])?;
        let list_reg = self.builder.assign_new(list);
        self.call_builtin("__make_reader_vector_from_list", vec![list_reg])
    }

    // __make_reader_map_0 - create an empty ReaderMap
    fn compile_builtin_make_reader_map_0(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        if !args.is_empty() {
            return Err(format!("__make_reader_map_0 takes no arguments, got {}", args.len()));
        }
        self.call_builtin("__make_reader_map_0", vec![])
    }

    // __reader_cons - prepend element to ReaderList
    fn compile_builtin_reader_cons(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        if args.len() != 2 {
            return Err(format!("__reader_cons requires 2 arguments, got {}", args.len()));
        }
        let elem = self.compile(&args[0])?;
        let list = self.compile(&args[1])?;
        let elem_reg = self.builder.assign_new(elem);
        let list_reg = self.builder.assign_new(list);
        self.call_builtin("__reader_cons", vec![elem_reg, list_reg])
    }

    // __reader_list_N - create ReaderList with N elements
    fn compile_builtin_reader_list_n(&mut self, args: &[Expr], n: usize) -> Result<IrValue, String> {
        if args.len() != n {
            return Err(format!("__reader_list_{} requires {} arguments, got {}", n, n, args.len()));
        }
        let mut arg_regs = Vec::with_capacity(n);
        for arg in args {
            let val = self.compile(arg)?;
            arg_regs.push(self.builder.assign_new(val));
        }
        self.call_builtin(&format!("__reader_list_{}", n), arg_regs)
    }

    /// Compile 1-arg reader primitive operations (first, rest, count)
    fn compile_builtin_reader_prim_1(&mut self, args: &[Expr], builtin_name: &str) -> Result<IrValue, String> {
        if args.len() != 1 {
            return Err(format!("{} requires 1 argument, got {}", builtin_name, args.len()));
        }
        let val = self.compile(&args[0])?;
        let val_reg = self.builder.assign_new(val);
        self.call_builtin(builtin_name, vec![val_reg])
    }

    /// Compile 2-arg reader primitive operations (nth, conj)
    fn compile_builtin_reader_prim_2(&mut self, args: &[Expr], builtin_name: &str) -> Result<IrValue, String> {
        if args.len() != 2 {
            return Err(format!("{} requires 2 arguments, got {}", builtin_name, args.len()));
        }
        let val1 = self.compile(&args[0])?;
        let val2 = self.compile(&args[1])?;
        let val1_reg = self.builder.assign_new(val1);
        let val2_reg = self.builder.assign_new(val2);
        self.call_builtin(builtin_name, vec![val1_reg, val2_reg])
    }

    /// Compile __gensym_0 (no arguments)
    fn compile_builtin_gensym_0(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        if !args.is_empty() {
            return Err(format!("__gensym_0 requires 0 arguments, got {}", args.len()));
        }
        self.call_builtin("__gensym_0", vec![])
    }

    fn compile_builtin_eq(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        if args.len() != 2 {
            return Err(format!("= requires 2 arguments, got {}", args.len()));
        }

        let left = self.compile(&args[0])?;
        let right = self.compile(&args[1])?;

        // Ensure operands are in registers (convert constants if needed)
        let left = self.builder.assign_new(left);
        let right = self.builder.assign_new(right);

        // Compare tagged values directly - this preserves type information
        // nil (7), false (3), true (11), and integers (n<<3) all have different representations
        let result = self.builder.new_register();
        self.builder
            .emit(Instruction::Compare(result, left, right, Condition::Equal));

        // Compare returns properly tagged boolean (3 or 11)
        Ok(result)
    }

    fn compile_builtin_gc(&mut self, _args: &[Expr]) -> Result<IrValue, String> {
        // __gc takes no arguments and returns nil
        // Call builtin_gc(frame_pointer) -> nil
        let result = self.builder.new_register();
        let builtin_addr = crate::trampoline::builtin_gc as usize;
        // Use FramePointer (x29) for GC stack walking. gc_return_addr is computed internally.
        self.builder.emit(Instruction::ExternalCall(
            result,
            builtin_addr,
            vec![IrValue::FramePointer],
        ));
        Ok(result)
    }

    // Array operations - direct calls to builtin functions
    // Use assign_new() to ensure all args are in registers before ExternalCall

    fn compile_builtin_make_array(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        if args.len() != 1 {
            return Err(format!("make-array requires 1 argument, got {}", args.len()));
        }
        let length = self.compile(&args[0])?;
        let length_reg = self.builder.assign_new(length);
        let result = self.builder.new_register();
        let builtin_addr = crate::trampoline::builtin_make_array as usize;
        // Allocating operation - pass FramePointer for GC. gc_return_addr is computed internally.
        self.builder.emit(Instruction::ExternalCall(
            result,
            builtin_addr,
            vec![IrValue::FramePointer, length_reg],
        ));
        Ok(result)
    }

    fn compile_builtin_aget(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        if args.len() != 2 {
            return Err(format!("aget requires 2 arguments, got {}", args.len()));
        }
        let array = self.compile(&args[0])?;
        let index = self.compile(&args[1])?;
        let array_reg = self.builder.assign_new(array);
        let index_reg = self.builder.assign_new(index);
        let result = self.builder.new_register();
        let builtin_addr = crate::trampoline::builtin_aget as usize;
        // Non-allocating - no need for FramePointer/ReturnAddress
        self.builder.emit(Instruction::ExternalCall(
            result,
            builtin_addr,
            vec![array_reg, index_reg],
        ));
        Ok(result)
    }

    fn compile_builtin_aset(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        if args.len() != 3 {
            return Err(format!("aset requires 3 arguments, got {}", args.len()));
        }
        let array = self.compile(&args[0])?;
        let index = self.compile(&args[1])?;
        let value = self.compile(&args[2])?;
        let array_reg = self.builder.assign_new(array);
        let index_reg = self.builder.assign_new(index);
        let value_reg = self.builder.assign_new(value);
        let result = self.builder.new_register();
        let builtin_addr = crate::trampoline::builtin_aset as usize;
        // Non-allocating - no need for FramePointer/ReturnAddress
        self.builder.emit(Instruction::ExternalCall(
            result,
            builtin_addr,
            vec![array_reg, index_reg, value_reg],
        ));
        Ok(result)
    }

    fn compile_builtin_alength(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        if args.len() != 1 {
            return Err(format!("alength requires 1 argument, got {}", args.len()));
        }
        let array = self.compile(&args[0])?;
        let array_reg = self.builder.assign_new(array);
        let result = self.builder.new_register();
        let builtin_addr = crate::trampoline::builtin_alength as usize;
        // Non-allocating - no need for FramePointer/ReturnAddress
        self.builder.emit(Instruction::ExternalCall(
            result,
            builtin_addr,
            vec![array_reg],
        ));
        Ok(result)
    }

    fn compile_builtin_aclone(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        if args.len() != 1 {
            return Err(format!("aclone requires 1 argument, got {}", args.len()));
        }
        let array = self.compile(&args[0])?;
        let array_reg = self.builder.assign_new(array);
        let result = self.builder.new_register();
        let builtin_addr = crate::trampoline::builtin_aclone as usize;
        // Allocating operation - pass FramePointer for GC. gc_return_addr is computed internally.
        self.builder.emit(Instruction::ExternalCall(
            result,
            builtin_addr,
            vec![IrValue::FramePointer, array_reg],
        ));
        Ok(result)
    }

    /// Compile __keyword_name (1 argument, allocating)
    fn compile_builtin_keyword_name(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        if args.len() != 1 {
            return Err(format!("__keyword_name requires 1 argument, got {}", args.len()));
        }
        let kw = self.compile(&args[0])?;
        let kw_reg = self.builder.assign_new(kw);
        let result = self.builder.new_register();
        let builtin_addr = crate::builtins::builtin__keyword_name as usize;
        // Allocating operation - pass FramePointer for GC
        self.builder.emit(Instruction::ExternalCall(
            result,
            builtin_addr,
            vec![IrValue::FramePointer, kw_reg],
        ));
        Ok(result)
    }

    /// Compile __apply (2 arguments, allocating)
    fn compile_builtin_apply(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        if args.len() != 2 {
            return Err(format!("__apply requires 2 arguments, got {}", args.len()));
        }
        let fn_val = self.compile(&args[0])?;
        let args_val = self.compile(&args[1])?;
        let fn_reg = self.builder.assign_new(fn_val);
        let args_reg = self.builder.assign_new(args_val);
        let result = self.builder.new_register();
        let builtin_addr = crate::builtins::builtin__apply as usize;
        // Allocating operation - pass FramePointer for GC
        self.builder.emit(Instruction::ExternalCall(
            result,
            builtin_addr,
            vec![IrValue::FramePointer, fn_reg, args_reg],
        ));
        Ok(result)
    }

    /// Compile __str_concat (1 argument - the seq of values, allocating)
    fn compile_builtin_str_concat(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        if args.len() != 1 {
            return Err(format!("__str_concat requires 1 argument, got {}", args.len()));
        }
        let seq = self.compile(&args[0])?;
        let seq_reg = self.builder.assign_new(seq);
        let result = self.builder.new_register();
        let builtin_addr = crate::builtins::builtin__str_concat as usize;
        // Allocating operation - pass FramePointer for GC
        self.builder.emit(Instruction::ExternalCall(
            result,
            builtin_addr,
            vec![IrValue::FramePointer, seq_reg],
        ));
        Ok(result)
    }

    /// Compile __reader_set_conj (2 arguments)
    fn compile_builtin_reader_set_conj(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        // Just use the standard 2-arg pattern
        self.compile_builtin_reader_prim_2(args, "__reader_set_conj")
    }

    // Bitwise operations - all work on untagged integers:
    // 1. Untag operands (shift right 3 bits)
    // 2. Perform bitwise operation
    // 3. Re-tag result (shift left 3 bits)

    fn compile_builtin_bit_and(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        if args.len() != 2 {
            return Err(format!("bit-and requires 2 arguments, got {}", args.len()));
        }

        let left = self.compile(&args[0])?;
        let right = self.compile(&args[1])?;

        let left_untagged = self.builder.new_register();
        let right_untagged = self.builder.new_register();
        self.builder.emit(Instruction::Untag(left_untagged, left));
        self.builder.emit(Instruction::Untag(right_untagged, right));

        let result_untagged = self.builder.new_register();
        self.builder.emit(Instruction::BitAnd(
            result_untagged,
            left_untagged,
            right_untagged,
        ));

        let result = self.builder.new_register();
        self.builder.emit(Instruction::Tag(
            result,
            result_untagged,
            IrValue::TaggedConstant(0),
        ));

        Ok(result)
    }

    fn compile_builtin_bit_or(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        if args.len() != 2 {
            return Err(format!("bit-or requires 2 arguments, got {}", args.len()));
        }

        let left = self.compile(&args[0])?;
        let right = self.compile(&args[1])?;

        let left_untagged = self.builder.new_register();
        let right_untagged = self.builder.new_register();
        self.builder.emit(Instruction::Untag(left_untagged, left));
        self.builder.emit(Instruction::Untag(right_untagged, right));

        let result_untagged = self.builder.new_register();
        self.builder.emit(Instruction::BitOr(
            result_untagged,
            left_untagged,
            right_untagged,
        ));

        let result = self.builder.new_register();
        self.builder.emit(Instruction::Tag(
            result,
            result_untagged,
            IrValue::TaggedConstant(0),
        ));

        Ok(result)
    }

    fn compile_builtin_bit_xor(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        if args.len() != 2 {
            return Err(format!("bit-xor requires 2 arguments, got {}", args.len()));
        }

        let left = self.compile(&args[0])?;
        let right = self.compile(&args[1])?;

        let left_untagged = self.builder.new_register();
        let right_untagged = self.builder.new_register();
        self.builder.emit(Instruction::Untag(left_untagged, left));
        self.builder.emit(Instruction::Untag(right_untagged, right));

        let result_untagged = self.builder.new_register();
        self.builder.emit(Instruction::BitXor(
            result_untagged,
            left_untagged,
            right_untagged,
        ));

        let result = self.builder.new_register();
        self.builder.emit(Instruction::Tag(
            result,
            result_untagged,
            IrValue::TaggedConstant(0),
        ));

        Ok(result)
    }

    fn compile_builtin_bit_not(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        if args.len() != 1 {
            return Err(format!("bit-not requires 1 argument, got {}", args.len()));
        }

        let operand = self.compile(&args[0])?;

        let operand_untagged = self.builder.new_register();
        self.builder
            .emit(Instruction::Untag(operand_untagged, operand));

        let result_untagged = self.builder.new_register();
        self.builder
            .emit(Instruction::BitNot(result_untagged, operand_untagged));

        let result = self.builder.new_register();
        self.builder.emit(Instruction::Tag(
            result,
            result_untagged,
            IrValue::TaggedConstant(0),
        ));

        Ok(result)
    }

    fn compile_builtin_bit_shift_left(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        if args.len() != 2 {
            return Err(format!(
                "bit-shift-left requires 2 arguments, got {}",
                args.len()
            ));
        }

        let value = self.compile(&args[0])?;
        let amount = self.compile(&args[1])?;

        let value_untagged = self.builder.new_register();
        let amount_untagged = self.builder.new_register();
        self.builder.emit(Instruction::Untag(value_untagged, value));
        self.builder
            .emit(Instruction::Untag(amount_untagged, amount));

        let result_untagged = self.builder.new_register();
        self.builder.emit(Instruction::BitShiftLeft(
            result_untagged,
            value_untagged,
            amount_untagged,
        ));

        let result = self.builder.new_register();
        self.builder.emit(Instruction::Tag(
            result,
            result_untagged,
            IrValue::TaggedConstant(0),
        ));

        Ok(result)
    }

    fn compile_builtin_bit_shift_right(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        if args.len() != 2 {
            return Err(format!(
                "bit-shift-right requires 2 arguments, got {}",
                args.len()
            ));
        }

        let value = self.compile(&args[0])?;
        let amount = self.compile(&args[1])?;

        let value_untagged = self.builder.new_register();
        let amount_untagged = self.builder.new_register();
        self.builder.emit(Instruction::Untag(value_untagged, value));
        self.builder
            .emit(Instruction::Untag(amount_untagged, amount));

        let result_untagged = self.builder.new_register();
        self.builder.emit(Instruction::BitShiftRight(
            result_untagged,
            value_untagged,
            amount_untagged,
        ));

        let result = self.builder.new_register();
        self.builder.emit(Instruction::Tag(
            result,
            result_untagged,
            IrValue::TaggedConstant(0),
        ));

        Ok(result)
    }

    fn compile_builtin_unsigned_bit_shift_right(
        &mut self,
        args: &[Expr],
    ) -> Result<IrValue, String> {
        if args.len() != 2 {
            return Err(format!(
                "unsigned-bit-shift-right requires 2 arguments, got {}",
                args.len()
            ));
        }

        let value = self.compile(&args[0])?;
        let amount = self.compile(&args[1])?;

        let value_untagged = self.builder.new_register();
        let amount_untagged = self.builder.new_register();
        self.builder.emit(Instruction::Untag(value_untagged, value));
        self.builder
            .emit(Instruction::Untag(amount_untagged, amount));

        let result_untagged = self.builder.new_register();
        self.builder.emit(Instruction::UnsignedBitShiftRight(
            result_untagged,
            value_untagged,
            amount_untagged,
        ));

        let result = self.builder.new_register();
        self.builder.emit(Instruction::Tag(
            result,
            result_untagged,
            IrValue::TaggedConstant(0),
        ));

        Ok(result)
    }

    // Type predicates

    fn compile_builtin_nil_pred(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        if args.len() != 1 {
            return Err(format!("nil? requires 1 argument, got {}", args.len()));
        }
        let val = self.compile(&args[0])?;
        let val = self.ensure_register(val);

        // Load nil constant (7) into a register for comparison
        let nil_const = self.builder.new_register();
        self.builder
            .emit(Instruction::LoadConstant(nil_const, IrValue::Null));

        let result = self.builder.new_register();
        self.builder.emit(Instruction::Compare(
            result,
            val,
            nil_const,
            Condition::Equal,
        ));
        Ok(result)
    }

    fn compile_builtin_number_pred(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        if args.len() != 1 {
            return Err(format!("number? requires 1 argument, got {}", args.len()));
        }
        let val = self.compile(&args[0])?;
        let val = self.ensure_register(val);
        let tag = self.builder.new_register();
        self.builder.emit(Instruction::GetTag(tag, val));

        // Load constant 2 into a register for comparison
        let two_const = self.builder.new_register();
        self.builder.emit(Instruction::LoadConstant(
            two_const,
            IrValue::TaggedConstant(2),
        ));

        // Check if tag < 2 (i.e., tag is 0 for int or 1 for float)
        let result = self.builder.new_register();
        self.builder.emit(Instruction::Compare(
            result,
            tag,
            two_const,
            Condition::LessThan,
        ));
        Ok(result)
    }

    fn compile_builtin_string_pred(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        if args.len() != 1 {
            return Err(format!("string? requires 1 argument, got {}", args.len()));
        }
        let val = self.compile(&args[0])?;
        let val = self.ensure_register(val);
        let tag = self.builder.new_register();
        self.builder.emit(Instruction::GetTag(tag, val));

        // Load constant 2 into a register for comparison
        let two_const = self.builder.new_register();
        self.builder.emit(Instruction::LoadConstant(
            two_const,
            IrValue::TaggedConstant(2),
        ));

        let result = self.builder.new_register();
        self.builder.emit(Instruction::Compare(
            result,
            tag,
            two_const,
            Condition::Equal,
        ));
        Ok(result)
    }

    fn compile_builtin_fn_pred(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        if args.len() != 1 {
            return Err(format!("fn? requires 1 argument, got {}", args.len()));
        }
        let val = self.compile(&args[0])?;
        let val = self.ensure_register(val);
        let tag = self.builder.new_register();
        self.builder.emit(Instruction::GetTag(tag, val));

        // fn? is true if tag == 4 (function) or tag == 5 (closure)
        // Use branching: check tag == 4, if not check tag == 5
        let is_fn_label = self.builder.new_label();
        let done_label = self.builder.new_label();
        let result = self.builder.new_register();

        // Load constants into registers
        let four_const = self.builder.new_register();
        self.builder.emit(Instruction::LoadConstant(
            four_const,
            IrValue::TaggedConstant(4),
        ));
        let five_const = self.builder.new_register();
        self.builder.emit(Instruction::LoadConstant(
            five_const,
            IrValue::TaggedConstant(5),
        ));

        // Check tag == 4 (function)
        self.builder.emit(Instruction::JumpIf(
            is_fn_label.clone(),
            Condition::Equal,
            tag,
            four_const,
        ));

        // Check tag == 5 (closure)
        self.builder.emit(Instruction::JumpIf(
            is_fn_label.clone(),
            Condition::Equal,
            tag,
            five_const,
        ));

        // Neither: return false
        self.builder.emit(Instruction::LoadFalse(result));
        self.builder.emit(Instruction::Jump(done_label.clone()));

        // Is function: return true
        self.builder.emit(Instruction::Label(is_fn_label));
        self.builder.emit(Instruction::LoadTrue(result));

        self.builder.emit(Instruction::Label(done_label));
        Ok(result)
    }

    fn compile_builtin_identical(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        if args.len() != 2 {
            return Err(format!(
                "identical? requires 2 arguments, got {}",
                args.len()
            ));
        }
        let left = self.compile(&args[0])?;
        let left = self.ensure_register(left);
        let right = self.compile(&args[1])?;
        let right = self.ensure_register(right);

        // Compare raw tagged values - identical values have identical bit patterns
        let result = self.builder.new_register();
        self.builder
            .emit(Instruction::Compare(result, left, right, Condition::Equal));
        Ok(result)
    }

    /// Compile (instance? Type obj) - check if obj is an instance of Type
    /// Type must be a symbol that was defined via deftype
    fn compile_builtin_instance_check(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        use crate::gc_runtime::DEFTYPE_ID_OFFSET;

        if args.len() != 2 {
            return Err(format!(
                "instance? requires 2 arguments, got {}",
                args.len()
            ));
        }

        // First argument must be a type symbol (e.g., PersistentVector)
        let type_name = match &args[0] {
            Expr::Var { namespace, name } => {
                // Build fully qualified type name
                let current_ns = self.get_current_namespace();
                match namespace {
                    Some(ns) => format!("{}/{}", ns, name),
                    None => format!("{}/{}", current_ns, name),
                }
            }
            _ => return Err("instance? first argument must be a type symbol".to_string()),
        };

        // Look up the type ID in the runtime
        let full_type_id = {
            let rt = unsafe { &*self.runtime.get() };
            match rt.get_type_id(&type_name) {
                Some(id) => id + DEFTYPE_ID_OFFSET,
                None => return Err(format!("Unknown type: {}", type_name)),
            }
        };

        // Compile the value to check
        let value = self.compile(&args[1])?;
        let value = self.ensure_register(value);

        // Emit instance check as ExternalCall
        // Call builtin_instance_check(type_id, value) -> tagged_boolean
        let result = self.builder.new_register();
        let builtin_addr = crate::trampoline::builtin_instance_check as usize;
        self.builder.emit(Instruction::ExternalCall(
            result,
            builtin_addr,
            vec![IrValue::RawConstant(full_type_id as i64), value],
        ));
        Ok(result)
    }

    /// Compile (satisfies? Protocol obj) - check if obj satisfies Protocol
    /// Protocol must be a symbol that was defined via defprotocol
    fn compile_builtin_satisfies_check(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        if args.len() != 2 {
            return Err(format!(
                "satisfies? requires 2 arguments, got {}",
                args.len()
            ));
        }

        // First argument must be a protocol symbol (e.g., IList, ISeq)
        let protocol_name = match &args[0] {
            Expr::Var { namespace, name } => {
                // Build fully qualified protocol name
                let current_ns = self.get_current_namespace();
                match namespace {
                    Some(ns) => format!("{}/{}", ns, name),
                    None => format!("{}/{}", current_ns, name),
                }
            }
            _ => return Err("satisfies? first argument must be a protocol symbol".to_string()),
        };

        // Look up the protocol ID in the runtime
        let protocol_id = {
            let rt = unsafe { &*self.runtime.get() };
            match rt.get_protocol_id(&protocol_name) {
                Some(id) => id,
                None => return Err(format!("Unknown protocol: {}", protocol_name)),
            }
        };

        // Compile the value to check
        let value = self.compile(&args[1])?;
        let value = self.ensure_register(value);

        // Emit satisfies check as ExternalCall
        // Call builtin_satisfies(protocol_id, value) -> tagged_boolean
        let result = self.builder.new_register();
        let fn_addr = crate::builtins::builtin_satisfies as usize;
        self.builder.emit(Instruction::ExternalCall(
            result,
            fn_addr,
            vec![IrValue::RawConstant(protocol_id as i64), value],
        ));
        Ok(result)
    }

    /// Compile (keyword? x) - check if x is a keyword
    fn compile_builtin_keyword_check(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        if args.len() != 1 {
            return Err(format!("keyword? requires 1 argument, got {}", args.len()));
        }

        // Compile the value to check
        let value = self.compile(&args[0])?;
        let value = self.ensure_register(value);

        // Call builtin_is_keyword
        let result = self.builder.new_register();
        let fn_addr = crate::trampoline::builtin_is_keyword as usize;
        self.builder
            .emit(Instruction::ExternalCall(result, fn_addr, vec![value]));
        Ok(result)
    }

    /// Compile (hash-primitive x) - hash a primitive value (keyword, string, number)
    fn compile_builtin_hash_primitive(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        if args.len() != 1 {
            return Err(format!(
                "hash-primitive requires 1 argument, got {}",
                args.len()
            ));
        }

        // Compile the value to hash
        let value = self.compile(&args[0])?;
        let value = self.ensure_register(value);

        // Call builtin_hash_value
        let result = self.builder.new_register();
        let fn_addr = crate::trampoline::builtin_hash_value as usize;
        self.builder
            .emit(Instruction::ExternalCall(result, fn_addr, vec![value]));
        Ok(result)
    }

    /// Compile (cons? x) - check if x is a cons cell
    fn compile_builtin_cons_check(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        if args.len() != 1 {
            return Err(format!("cons? requires 1 argument, got {}", args.len()));
        }

        let value = self.compile(&args[0])?;
        let value = self.ensure_register(value);

        let result = self.builder.new_register();
        let fn_addr = crate::trampoline::builtin_is_cons as usize;
        self.builder
            .emit(Instruction::ExternalCall(result, fn_addr, vec![value]));
        Ok(result)
    }

    /// Compile (cons-first x) - get the first element of a cons cell
    fn compile_builtin_cons_first(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        if args.len() != 1 {
            return Err(format!(
                "cons-first requires 1 argument, got {}",
                args.len()
            ));
        }

        let value = self.compile(&args[0])?;
        let value = self.ensure_register(value);

        let result = self.builder.new_register();
        let fn_addr = crate::trampoline::builtin_cons_first as usize;
        self.builder
            .emit(Instruction::ExternalCall(result, fn_addr, vec![value]));
        Ok(result)
    }

    /// Compile (cons-rest x) - get the rest of a cons cell
    fn compile_builtin_cons_rest(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        if args.len() != 1 {
            return Err(format!("cons-rest requires 1 argument, got {}", args.len()));
        }

        let value = self.compile(&args[0])?;
        let value = self.ensure_register(value);

        let result = self.builder.new_register();
        let fn_addr = crate::trampoline::builtin_cons_rest as usize;
        self.builder
            .emit(Instruction::ExternalCall(result, fn_addr, vec![value]));
        Ok(result)
    }

    /// Get the generated IR instructions without consuming the compiler
    /// This clears the instruction buffer, allowing the compiler to be reused
    pub fn take_instructions(&mut self) -> Vec<Instruction> {
        self.builder.take_instructions()
    }

    /// Ensure a value is in a register (convert constants to registers if needed)
    /// This is used for top-level compilation to ensure the result is always a register.
    /// Following Beagle's pattern: constants are converted to registers at point of use.
    pub fn ensure_register(&mut self, val: IrValue) -> IrValue {
        self.builder.assign_new(val)
    }

    /// Take compiled function IRs (for display purposes)
    /// Returns vector of (function_name, instructions) for each nested function
    pub fn take_compiled_function_irs(&mut self) -> Vec<(Option<String>, Vec<Instruction>)> {
        std::mem::take(&mut self.compiled_function_irs)
    }
}
