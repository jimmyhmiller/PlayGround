use crate::clojure_ast::{Expr, CatchClause};
use crate::value::Value;
use crate::ir::{Instruction, IrValue, IrBuilder, Condition, Label};
use crate::gc_runtime::GCRuntime;
use crate::arm_codegen::Arm64CodeGen;
use std::collections::HashMap;
use std::sync::Arc;
use std::cell::UnsafeCell;

/// Context for loop/recur
#[derive(Clone)]
struct LoopContext {
    label: Label,
    binding_registers: Vec<IrValue>,
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
    fields: Vec<(String, bool)>,  // (name, is_mutable)
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
    builder: IrBuilder,

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
            let (ns_var_ptr, _ns_var_id) = rt.allocate_var(core_ns_ptr, "*ns*", user_ns_ptr).unwrap();
            let new_core_ns_ptr = rt.namespace_add_binding(core_ns_ptr, "*ns*", ns_var_ptr).unwrap();

            // Mark *ns* as dynamic so it can be rebound later
            rt.mark_var_dynamic(ns_var_ptr);

            // Update core_ns_ptr if it changed (due to namespace re-allocation)
            let core_ns_ptr = new_core_ns_ptr;

            // Bootstrap builtin functions as Vars in clojure.core
            let core_ns_ptr = rt.bootstrap_builtins(core_ns_ptr).unwrap();

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
            Expr::Def { name, value, metadata } => self.compile_def(name, value, metadata),
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
            Expr::FieldSet { field, object, value } => self.compile_field_set(field, object, value),
            Expr::Throw { exception } => self.compile_throw(exception),
            Expr::Try { body, catches, finally } => self.compile_try(body, catches, finally),
            // Protocol system - will be implemented in Phase 4
            Expr::DefProtocol { name, methods } => self.compile_defprotocol(name, methods),
            Expr::ExtendType { type_name, implementations } => self.compile_extend_type(type_name, implementations),
            Expr::ProtocolCall { method_name, args } => self.compile_protocol_call(method_name, args),
            Expr::Debugger { expr } => self.compile_debugger(expr),
        }
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
                let float_ptr = rt.allocate_float(*f)
                    .map_err(|e| format!("Failed to allocate float: {}", e))?;

                // Load the tagged pointer as a constant
                self.builder.emit(Instruction::LoadConstant(
                    result,
                    IrValue::TaggedConstant(float_ptr as isize),
                ));
            }
            Value::String(s) => {
                // Allocate string at compile time
                // SAFETY: Single-threaded REPL
                let rt = unsafe { &mut *self.runtime.get() };
                let str_ptr = rt.allocate_string(s)
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

                // Emit LoadKeyword instruction - actual allocation happens at runtime
                self.builder.emit(Instruction::LoadKeyword(result, keyword_index));
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
            && let Some(register) = self.lookup_local(name) {
                // Local variable - just return the register directly
                return Ok(register);
            }

        // Not a local - look up as a global var
        // SAFETY: Single-threaded REPL - no concurrent access during compilation
        let rt = unsafe { &*self.runtime.get() };

        // Determine which namespace to look in
        let ns_ptr = if let Some(ns_name) = namespace {
            self.namespace_registry
                .get(ns_name)
                .copied()
                .ok_or_else(|| format!("Undefined namespace: {}", ns_name))?
        } else {
            // Try current namespace first
            if let Some(var_ptr) = rt.namespace_lookup(self.current_namespace_ptr, name) {
                // eprintln!("DEBUG compile_var (current ns): found var '{}' at {:x}", name, var_ptr);
                let var_id = rt.get_var_id(var_ptr).expect("var should have var_id");
                let result = self.builder.new_register();
                // Check if var is dynamic at compile time
                if rt.is_var_dynamic(var_ptr) {
                    // Dynamic var - use trampoline for binding lookup
                    self.builder.emit(Instruction::LoadVarDynamic(
                        result,
                        var_id,
                    ));
                } else {
                    // Non-dynamic var - direct memory load
                    self.builder.emit(Instruction::LoadVar(
                        result,
                        var_id,
                    ));
                }
                return Ok(result);
            }

            // Try used namespaces
            let current_ns_name = rt.namespace_name(self.current_namespace_ptr);
            if let Some(used) = self.used_namespaces.get(&current_ns_name) {
                for used_ns in used {
                    if let Some(&used_ns_ptr) = self.namespace_registry.get(used_ns)
                        && let Some(var_ptr) = rt.namespace_lookup(used_ns_ptr, name) {
                            // eprintln!("DEBUG compile_var (used ns {}): found var '{}' at {:x}", used_ns, name, var_ptr);
                            let var_id = rt.get_var_id(var_ptr).expect("var should have var_id");
                            let result = self.builder.new_register();
                            // Check if var is dynamic at compile time
                            if rt.is_var_dynamic(var_ptr) {
                                self.builder.emit(Instruction::LoadVarDynamic(
                                    result,
                                    var_id,
                                ));
                            } else {
                                self.builder.emit(Instruction::LoadVar(
                                    result,
                                    var_id,
                                ));
                            }
                            return Ok(result);
                        }
                }
            }

            return Err(format!("Undefined variable: {}", name));
        };

        // Look up in specified namespace
        if let Some(var_ptr) = rt.namespace_lookup(ns_ptr, name) {
            // eprintln!("DEBUG compile_var: found var '{}' at {:x}", name, var_ptr);
            let var_id = rt.get_var_id(var_ptr).expect("var should have var_id");
            let result = self.builder.new_register();
            // Check if var is dynamic at compile time
            if rt.is_var_dynamic(var_ptr) {
                // Dynamic var - use trampoline for binding lookup
                self.builder.emit(Instruction::LoadVarDynamic(
                    result,
                    var_id,
                ));
            } else {
                // Non-dynamic var - direct memory load
                self.builder.emit(Instruction::LoadVar(
                    result,
                    var_id,
                ));
            }
            Ok(result)
        } else {
            Err(format!("Undefined variable: {}/{}", namespace.as_ref().unwrap(), name))
        }
    }

    /// Check if a symbol is a built-in function
    fn is_builtin(&self, name: &str) -> bool {
        matches!(name, "+" | "-" | "*" | "/" | "<" | ">" | "<=" | ">=" | "=" | "__gc" |
                 "bit-and" | "bit-or" | "bit-xor" | "bit-not" |
                 "bit-shift-left" | "bit-shift-right" | "unsigned-bit-shift-right" |
                 "bit-shift-right-zero-fill" |  // CLJS alias for unsigned-bit-shift-right
                 "nil?" | "number?" | "string?" | "fn?" | "identical?" | "println")
    }

    /// Compile (var symbol) - returns the Var object itself, not its value
    fn compile_var_ref(&mut self, namespace: &Option<String>, name: &str) -> Result<IrValue, String> {
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
    fn compile_deftype(&mut self, name: &str, fields: &[crate::clojure_ast::FieldDef]) -> Result<IrValue, String> {
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
        self.builder.emit(Instruction::LoadConstant(result, IrValue::Null));
        Ok(result)
    }

    /// Compile (TypeName. arg1 arg2 ...) - constructor call
    fn compile_type_construct(&mut self, type_name: &str, args: &[Expr]) -> Result<IrValue, String> {
        // Resolve type name to fully qualified name
        let qualified_name = if type_name.contains('/') {
            // Already qualified (e.g., "foo/Point")
            type_name.to_string()
        } else {
            // Unqualified - use current namespace
            format!("{}/{}", self.get_current_namespace(), type_name)
        };

        // Look up the type by fully qualified name
        let type_info = self.type_registry.get(&qualified_name)
            .cloned()
            .ok_or_else(|| format!("Unknown type: {}", qualified_name))?;

        // Check arg count matches field count
        if args.len() != type_info.fields.len() {
            return Err(format!(
                "Type {} requires {} arguments, got {}",
                type_name, type_info.fields.len(), args.len()
            ));
        }

        // Compile all arguments
        let mut arg_values = Vec::new();
        for arg in args {
            arg_values.push(self.compile(arg)?);
        }

        // Emit MakeType instruction
        let result = self.builder.new_register();
        self.builder.emit(Instruction::MakeType(result, type_info.type_id, arg_values));

        Ok(result)
    }

    /// Compile (.-field obj) - field access
    fn compile_field_access(&mut self, field: &str, object: &Expr) -> Result<IrValue, String> {
        // Compile the object
        let obj_value = self.compile(object)?;

        // We need runtime type lookup to find the field index
        // For now, emit a LoadTypeField with field name (runtime will resolve)
        // This is a simplification - a real implementation would use inline caching

        // Get field index at runtime via trampoline
        let result = self.builder.new_register();
        self.builder.emit(Instruction::LoadTypeField(result, obj_value, field.to_string()));

        Ok(result)
    }

    /// Compile (set! (.-field obj) value) - field assignment
    /// Requires the field to be declared as ^:mutable in the deftype
    fn compile_field_set(&mut self, field: &str, object: &Expr, value: &Expr) -> Result<IrValue, String> {
        // 1. Compile the object
        let obj_value = self.compile(object)?;

        // 2. Compile the new value and ensure it's in a register
        // (StoreTypeField requires a register, not a TaggedConstant)
        let new_value = self.compile(value)?;
        let new_value = self.ensure_register(new_value);

        // 3. Emit GcAddRoot for write barrier (BEFORE the store)
        // This is critical for generational GC correctness:
        // When storing a young-generation pointer into an old-generation object,
        // we must track the old object so the GC can find the cross-generational reference.
        self.builder.emit(Instruction::GcAddRoot(obj_value));

        // 4. Emit the field store
        self.builder.emit(Instruction::StoreTypeField(obj_value, field.to_string(), new_value));

        // 5. set! returns the new value
        Ok(new_value)
    }

    fn compile_throw(&mut self, exception: &Expr) -> Result<IrValue, String> {
        // Compile the exception expression
        let exc_val = self.compile(exception)?;

        // Emit Throw instruction
        self.builder.emit(Instruction::Throw(exc_val));

        // Throw never returns, but return dummy nil for type consistency
        let dummy = self.builder.new_register();
        self.builder.emit(Instruction::LoadConstant(dummy, IrValue::Null));
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
        self.builder.emit(Instruction::Jump(after_catch_label.clone()));

        // 5. Catch block (throw jumps here with exception stored at stack slot)
        self.builder.emit(Instruction::Label(catch_label.clone()));

        // Load exception from the pre-allocated stack slot into exception_local register
        self.builder.emit(Instruction::LoadExceptionLocal(exception_local, exception_slot));

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
                arities: m.arities.iter().map(|a| a.len()).collect(),
            })
            .collect();

        // Fully qualify the protocol name
        let full_name = format!("{}/{}", self.get_current_namespace(), name);

        unsafe {
            let rt = &mut *self.runtime.get();
            rt.register_protocol(full_name, protocol_methods);
        }

        // defprotocol returns nil
        let result = self.builder.new_register();
        self.builder.emit(Instruction::LoadConstant(result, IrValue::Null));
        Ok(result)
    }

    fn compile_extend_type(
        &mut self,
        type_name: &str,
        implementations: &[crate::clojure_ast::ProtocolImpl],
    ) -> Result<IrValue, String> {
        use crate::gc_runtime::DEFTYPE_ID_OFFSET;
        use std::collections::HashMap;

        // Resolve type_id
        let type_id = self.resolve_type_id(type_name)?;

        // For each protocol implementation
        for impl_ in implementations {
            // Get protocol_id
            let protocol_full_name = format!("{}/{}", self.get_current_namespace(), &impl_.protocol_name);
            let protocol_id = unsafe {
                let rt = &*self.runtime.get();
                rt.get_protocol_id(&protocol_full_name)
                    .ok_or_else(|| format!("Unknown protocol: {}", impl_.protocol_name))?
            };

            // Group methods by name to handle multi-arity protocol methods
            // Multiple implementations with the same name but different param counts
            // should be combined into a single multi-arity function
            let mut methods_by_name: HashMap<String, Vec<&crate::clojure_ast::ProtocolMethodImpl>> = HashMap::new();
            for method in &impl_.methods {
                methods_by_name.entry(method.name.clone()).or_default().push(method);
            }

            // For each unique method name
            for (method_name, method_impls) in methods_by_name {
                let method_index = unsafe {
                    let rt = &*self.runtime.get();
                    rt.get_protocol_method_index(protocol_id, &method_name)
                        .ok_or_else(|| format!("Unknown method {} in protocol {}", method_name, impl_.protocol_name))?
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

                // Emit RegisterProtocolMethod instruction
                self.builder.emit(Instruction::RegisterProtocolMethod(
                    type_id,
                    protocol_id,
                    method_index,
                    fn_value,
                ));
            }
        }

        // Return nil
        let result = self.builder.new_register();
        self.builder.emit(Instruction::LoadConstant(result, IrValue::Null));
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
            "nil" | "Nil" => Ok(BUILTIN_TYPE_NIL),
            "Boolean" | "bool" => Ok(BUILTIN_TYPE_BOOL),
            "Long" | "Integer" | "int" => Ok(BUILTIN_TYPE_INT),
            "Double" | "Float" | "float" => Ok(BUILTIN_TYPE_FLOAT),
            "String" => Ok(BUILTIN_TYPE_STRING),
            "Keyword" => Ok(BUILTIN_TYPE_KEYWORD),
            "Symbol" => Ok(BUILTIN_TYPE_SYMBOL),
            "PersistentList" | "List" => Ok(BUILTIN_TYPE_LIST),
            "PersistentVector" | "Vector" => Ok(BUILTIN_TYPE_VECTOR),
            "PersistentHashMap" | "Map" => Ok(BUILTIN_TYPE_MAP),
            "PersistentHashSet" | "Set" => Ok(BUILTIN_TYPE_SET),
            _ => Err(format!("Unknown type: {}", type_name)),
        }
    }

    fn compile_protocol_call(
        &mut self,
        method_name: &str,
        args: &[Expr],
    ) -> Result<IrValue, String> {
        if args.is_empty() {
            return Err(format!("Protocol method {} requires at least 1 argument", method_name));
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
        let arg_values: Vec<_> = arg_values.into_iter()
            .map(|v| self.builder.assign_new(v))
            .collect();

        // Step 1: Call trampoline_protocol_lookup(target, method_ptr, method_len) -> fn_ptr
        let fn_ptr = self.builder.new_register();
        let trampoline_addr = crate::trampoline::trampoline_protocol_lookup as usize;
        self.builder.emit(Instruction::ExternalCall(
            fn_ptr,
            trampoline_addr,
            vec![
                arg_values[0].clone(),                      // target
                IrValue::RawConstant(method_ptr as i64),    // method_name_ptr
                IrValue::RawConstant(method_len as i64),    // method_name_len
            ],
        ));

        // Step 2: Call fn_ptr with all args (register allocator handles saves)
        let result = self.builder.new_register();
        self.builder.emit(Instruction::Call(result, fn_ptr, arg_values));

        Ok(result)
    }

    fn compile_debugger(&mut self, expr: &Expr) -> Result<IrValue, String> {
        // Emit breakpoint instruction before evaluating the expression
        self.builder.emit(Instruction::Breakpoint);
        // Compile and return the expression
        self.compile(expr)
    }

    fn compile_def(&mut self, name: &str, value_expr: &Expr, metadata: &Option<im::HashMap<String, Value>>) -> Result<IrValue, String> {
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

        // SAFETY: Single-threaded REPL - no concurrent access during compilation
        // IMPORTANT: Create/lookup the Var BEFORE compiling the value expression.
        // This allows recursive functions to resolve their own name via LoadVar.
        let (var_ptr, var_id) = unsafe {
            let rt = &mut *self.runtime.get();

            // Check if var already exists
            if let Some(existing_var_ptr) = rt.namespace_lookup(self.current_namespace_ptr, name) {
                // Get var_id for existing var
                let var_id = rt.get_var_id(existing_var_ptr).expect("existing var should have var_id");
                (existing_var_ptr, var_id)
            } else {
                // Create new var with nil placeholder (7) - will be overwritten after compilation
                let (new_var_ptr, new_var_id) = rt.allocate_var(
                    self.current_namespace_ptr,
                    name,
                    7, // nil placeholder
                ).unwrap();

                // Mark as dynamic ONLY if metadata indicates it
                if is_dynamic {
                    rt.mark_var_dynamic(new_var_ptr);
                }

                // Add var to namespace BEFORE compiling value
                // This enables recursive function references
                let new_ns_ptr = rt
                    .namespace_add_binding(self.current_namespace_ptr, name, new_var_ptr)
                    .unwrap();

                // Update current namespace pointer if it moved
                // (namespace_add_binding now handles updating the GC root automatically)
                if new_ns_ptr != self.current_namespace_ptr {
                    self.current_namespace_ptr = new_ns_ptr;

                    // Update in registry
                    let ns_name = rt.namespace_name(new_ns_ptr);
                    self.namespace_registry.insert(ns_name, new_ns_ptr);
                }

                (new_var_ptr, new_var_id)
            }
        };

        // NOW compile the value expression - recursive refs will resolve via LoadVar
        let value_reg = self.compile(value_expr)?;

        // Emit StoreVar instruction to update the var at runtime
        // Use var_id for GC-safe indirect access via var table
        self.builder.emit(Instruction::StoreVar(
            var_id,
            value_reg,
        ));

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

    /// Sync namespace registry with runtime after GC relocations
    pub fn sync_namespace_registry(&mut self) {
        // SAFETY: Single-threaded access after GC
        unsafe {
            let rt = &*self.runtime.get();

            // Find current namespace name by reverse lookup BEFORE updating registry
            // (Don't dereference current_namespace_ptr - it may be stale after compacting GC)
            let current_ns_name = self.namespace_registry
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
            if let Some(name) = current_ns_name {
                if let Some(&new_ptr) = rt.get_namespace_pointers().get(&name) {
                    self.current_namespace_ptr = new_ptr;
                }
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
                let ns_ptr = rt.allocate_namespace(name)?;
                rt.add_namespace_root(name.to_string(), ns_ptr);
                ns_ptr
            };

            self.namespace_registry.insert(name.to_string(), ns_ptr);
            self.current_namespace_ptr = ns_ptr;

            // Implicitly use clojure.core
            self.used_namespaces.insert(name.to_string(), vec!["clojure.core".to_string()]);
        }

        // Emit instruction to set *ns* at runtime
        // Look up the *ns* var pointer and var_id
        let ns_var_id = unsafe {
            let rt = &*self.runtime.get();
            let core_ns_ptr = *self.namespace_registry.get("clojure.core").unwrap();
            let ns_var_ptr = rt.namespace_lookup(core_ns_ptr, "*ns*").unwrap();
            rt.get_var_id(ns_var_ptr).expect("*ns* var should have var_id")
        };

        // Load the current namespace pointer as a tagged constant
        let ns_value = self.builder.new_register();
        self.builder.emit(Instruction::LoadConstant(
            ns_value,
            IrValue::TaggedConstant((self.current_namespace_ptr << 3) as isize)
        ));

        // Store it into *ns* var
        self.builder.emit(Instruction::StoreVar(
            ns_var_id,
            ns_value,
        ));

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
        self.builder.emit(Instruction::LoadConstant(result, IrValue::Null));
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
        self.builder.emit(Instruction::LoadConstant(nil_reg, IrValue::Null));
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
            self.builder.emit(Instruction::LoadConstant(result, IrValue::Null));
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

    fn compile_let(&mut self, bindings: &[(String, Box<Expr>)], body: &[Expr]) -> Result<IrValue, String> {
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

    fn compile_loop(&mut self, bindings: &[(String, Box<Expr>)], body: &[Expr]) -> Result<IrValue, String> {
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
        self.loop_contexts.push(LoopContext {
            label: loop_label,
            binding_registers,
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
        let context = self.loop_contexts.last()
            .ok_or_else(|| "recur not in loop or fn context".to_string())?
            .clone();

        if args.len() != context.binding_registers.len() {
            return Err(format!(
                "recur arity mismatch: expected {}, got {}",
                context.binding_registers.len(),
                args.len()
            ));
        }

        // Compile ALL new values first (before any assignments)
        let mut new_values = Vec::new();
        for arg in args {
            new_values.push(self.compile(arg)?);
        }

        // Build assignments list: (target_binding_register, new_value)
        let mut assignments = Vec::new();
        for (binding_reg, new_value) in context.binding_registers.iter().zip(new_values.iter()) {
            // Include all assignments, even if target == source
            // Codegen will skip no-ops
            assignments.push((*binding_reg, *new_value));
        }

        // Emit Recur instruction (will be transformed to RecurWithSaves by register allocator)
        // The Recur instruction handles: 1) saving live registers, 2) assignments, 3) jump
        self.builder.emit(Instruction::Recur(context.label.clone(), assignments));

        // Return dummy (recur never returns normally)
        let dummy = self.builder.new_register();
        self.builder.emit(Instruction::LoadConstant(dummy, IrValue::Null));
        Ok(dummy)
    }

    fn compile_binding(&mut self, bindings: &[(String, Box<Expr>)], body: &[Expr]) -> Result<IrValue, String> {
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

            // Emit PushBinding instruction
            self.builder.emit(Instruction::PushBinding(
                IrValue::TaggedConstant(var_ptr as isize),
                value_reg,
            ));

            var_ptrs.push(var_ptr);
        }

        // Compile body (like do - last expression's value is returned)
        let mut result = self.builder.new_register();
        self.builder.emit(Instruction::LoadConstant(result, IrValue::Null));

        for expr in body {
            result = self.compile(expr)?;
        }

        // Pop all bindings in reverse order
        for &var_ptr in var_ptrs.iter().rev() {
            self.builder.emit(Instruction::PopBinding(
                IrValue::TaggedConstant(var_ptr as isize)
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

        // Emit SetVar instruction
        self.builder.emit(Instruction::SetVar(
            IrValue::TaggedConstant(var_ptr as isize),
            value_reg,
        ));

        // set! returns the new value
        Ok(value_reg)
    }

    /// Parse a var name into namespace and name components
    /// Handles both qualified (ns/name) and unqualified (name) symbols
    fn parse_var_name(&self, var_name: &str) -> Result<(Option<String>, String), String> {
        if let Some(idx) = var_name.find('/') {
            let namespace = var_name[..idx].to_string();
            let name = var_name[idx+1..].to_string();
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
            *self.namespace_registry.get(ns_name)
                .ok_or_else(|| format!("Namespace not found: {}", ns_name))?
        } else {
            self.current_namespace_ptr
        };

        // Look up the var
        rt.namespace_lookup(ns_ptr, name)
            .ok_or_else(|| format!("Var not found: {}", name))
    }

    fn compile_fn(&mut self, name: &Option<String>, arities: &[crate::value::FnArity]) -> Result<IrValue, String> {
        // Per-function compilation (Beagle's approach):
        // Each function (or each arity) is compiled separately with its own register allocation

        // Step 1: Analyze closures - find free variables across ALL arities (union)
        let free_vars = self.find_free_variables_across_arities(arities, name)?;

        // DEBUG: Print free vars for protocol methods
        if !free_vars.is_empty() {
            eprintln!("DEBUG compile_fn: name={:?}, free_vars={:?}", name, free_vars);
        }

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

            let code_ptr = self.compile_single_arity(name, arity, &free_vars, is_multi_arity, arity_count)?;

            compiled_arities.push((param_count, code_ptr, is_variadic));
        }

        // Step 4: Determine variadic minimum (if any)
        let variadic_min = compiled_arities.iter()
            .find(|(_, _, is_var)| *is_var)
            .map(|(count, _, _)| *count);

        // Step 5: Create function object
        let fn_obj_reg = self.builder.new_register();

        if arities.len() == 1 && variadic_min.is_none() {
            // Single fixed arity - use existing MakeFunctionPtr (more efficient)
            self.builder.emit(Instruction::MakeFunctionPtr(
                fn_obj_reg,
                compiled_arities[0].1,
                closure_values,
            ));
        } else {
            // Multi-arity or variadic - use MakeMultiArityFn
            let arity_table: Vec<(usize, usize)> = compiled_arities.iter()
                .map(|(param_count, code_ptr, _)| (*param_count, *code_ptr))
                .collect();

            self.builder.emit(Instruction::MakeMultiArityFn(
                fn_obj_reg,
                arity_table,
                variadic_min,
                closure_values,
            ));
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
        let outer_builder = std::mem::replace(&mut self.builder, IrBuilder::new());

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
        for (i, param) in arity.params.iter().enumerate() {
            let arg_reg = IrValue::Register(crate::ir::VirtualRegister::Argument(i + param_offset));
            // Allocate a local slot and store the argument there
            let local_slot = self.builder.allocate_local();
            self.builder.emit(Instruction::StoreLocal(local_slot, arg_reg));
            self.bind_local_slot(param.clone(), local_slot);
            // For recur, we still need a register representation
            // Load the parameter into a temp register for the param_registers vec
            let temp_reg = self.builder.new_register();
            self.builder.emit(Instruction::LoadLocal(temp_reg, local_slot));
            param_registers.push(temp_reg);
        }

        // Bind rest parameter if variadic
        // CollectRestArgs reads the arg count from x9 (set by caller) and builds a list
        // from excess arguments beyond the fixed params
        if let Some(rest) = &arity.rest_param {
            let rest_reg = self.builder.new_register();
            let fixed_count = arity.params.len();
            self.builder.emit(Instruction::CollectRestArgs(rest_reg, fixed_count, param_offset));
            self.bind_local(rest.clone(), rest_reg);
            param_registers.push(rest_reg);
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
                        closure_reg, self_reg.clone(), arity_count, i
                    ));
                } else {
                    // Single-arity: use standard closure layout
                    self.builder.emit(Instruction::LoadClosure(closure_reg, self_reg.clone(), i));
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
        self.builder.emit(Instruction::Label(fn_entry_label.clone()));
        self.loop_contexts.push(LoopContext {
            label: fn_entry_label,
            binding_registers: param_registers,
        });

        // Compile body (implicit do)
        let result = self.compile_body(&arity.body)?;

        // Pop fn's loop context
        self.loop_contexts.pop();

        // Compile and check post-conditions
        // Each :post condition can reference % which is bound to the result
        if !arity.post_conditions.is_empty() {
            // Bind % to the result for post-condition expressions
            self.bind_local("%".to_string(), result.clone());

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
        self.compiled_function_irs.push((name.clone(), fn_instructions.clone()));

        // Get var_table_ptr for GC-safe var access in codegen
        let var_table_ptr = unsafe {
            let rt = &*self.runtime.get();
            rt.var_table_ptr() as usize
        };

        // Pass num_locals from the builder (includes all allocated local slots)
        let num_locals = self.builder.num_locals;
        let compiled = Arm64CodeGen::compile_function(&fn_instructions, num_locals, reserved_exception_slots, var_table_ptr)?;

        // Register stack map with runtime for GC
        unsafe {
            let rt = &mut *self.runtime.get();
            for (pc, stack_size) in &compiled.stack_map {
                rt.add_stack_map_entry(*pc, crate::gc::StackMapDetails {
                    function_name: name.clone(),
                    number_of_locals: compiled.num_locals,
                    current_stack_size: *stack_size,
                    max_stack_size: compiled.max_stack_size,
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
            self.builder.emit(Instruction::LoadConstant(result, IrValue::Null));
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
            Expr::Var { namespace: None, name } => {
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

            Expr::Fn { name: fn_name, arities } => {
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
            Expr::Var { namespace: Some(_), .. } => {}

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

            // Throw: check exception expression for free vars
            Expr::Throw { exception } => {
                self.collect_free_vars_from_expr(exception, bound, free);
            }

            // Try: check body, catches, and finally for free vars
            Expr::Try { body, catches, finally } => {
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

            Expr::ExtendType { implementations, .. } => {
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
                    "+" => self.compile_builtin_add(args),
                    "-" => self.compile_builtin_sub(args),
                    "*" => self.compile_builtin_mul(args),
                    "/" => self.compile_builtin_div(args),
                    "<" => self.compile_builtin_lt(args),
                    ">" => self.compile_builtin_gt(args),
                    "<=" => self.compile_builtin_le(args),
                    ">=" => self.compile_builtin_ge(args),
                    "=" => self.compile_builtin_eq(args),
                    "println" => self.compile_builtin_println(args),
                    "__gc" => self.compile_builtin_gc(args),
                    "bit-and" => self.compile_builtin_bit_and(args),
                    "bit-or" => self.compile_builtin_bit_or(args),
                    "bit-xor" => self.compile_builtin_bit_xor(args),
                    "bit-not" => self.compile_builtin_bit_not(args),
                    "bit-shift-left" => self.compile_builtin_bit_shift_left(args),
                    "bit-shift-right" => self.compile_builtin_bit_shift_right(args),
                    "unsigned-bit-shift-right" => self.compile_builtin_unsigned_bit_shift_right(args),
                    "bit-shift-right-zero-fill" => self.compile_builtin_unsigned_bit_shift_right(args),  // CLJS alias
                    "nil?" => self.compile_builtin_nil_pred(args),
                    "number?" => self.compile_builtin_number_pred(args),
                    "string?" => self.compile_builtin_string_pred(args),
                    "fn?" => self.compile_builtin_fn_pred(args),
                    "identical?" => self.compile_builtin_identical(args),
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
        let arg_vals: Vec<_> = arg_vals.into_iter()
            .map(|v| self.builder.assign_new(v))
            .collect();

        // 4. Emit Call instruction
        let result = self.builder.new_register();
        self.builder.emit(Instruction::Call(result, fn_val, arg_vals));

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
        self.builder.emit(int_op(int_result, left_untagged, right_untagged));

        // Tag as int (tag 0b000)
        self.builder.emit(Instruction::Tag(result, int_result, IrValue::TaggedConstant(0)));
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
        self.builder.emit(Instruction::LoadConstant(left_float, IrValue::TaggedConstant(0)));
        self.builder.emit(Instruction::IntToFloat(left_float, left_float));
        self.builder.emit(Instruction::Jump(left_convert_done.clone()));

        // Left is int, convert to float
        self.builder.emit(Instruction::Label(left_is_int));
        let left_int_untagged = self.builder.new_register();
        self.builder.emit(Instruction::Untag(left_int_untagged, left));
        self.builder.emit(Instruction::IntToFloat(left_float, left_int_untagged));
        self.builder.emit(Instruction::Jump(left_convert_done.clone()));

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
        self.builder.emit(Instruction::LoadConstant(right_float, IrValue::TaggedConstant(0)));
        self.builder.emit(Instruction::IntToFloat(right_float, right_float));
        self.builder.emit(Instruction::Jump(right_convert_done.clone()));

        // Right is int, convert to float
        self.builder.emit(Instruction::Label(right_is_int));
        let right_int_untagged = self.builder.new_register();
        self.builder.emit(Instruction::Untag(right_int_untagged, right));
        self.builder.emit(Instruction::IntToFloat(right_float, right_int_untagged));
        self.builder.emit(Instruction::Jump(right_convert_done.clone()));

        // Right is already float, load f64 bits from heap
        self.builder.emit(Instruction::Label(right_is_float));
        self.builder.emit(Instruction::LoadFloat(right_float, right));

        self.builder.emit(Instruction::Label(right_convert_done));

        // Perform float operation (on raw f64 bits)
        let float_result = self.builder.new_register();
        self.builder.emit(float_op(float_result, left_float, right_float));

        // Allocate new float on heap and get tagged pointer
        self.builder.emit(Instruction::AllocateFloat(result, float_result));

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
        self.builder.emit(Instruction::Compare(result, left_untagged, right_untagged, Condition::LessThan));

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
        self.builder.emit(Instruction::Compare(result, left_untagged, right_untagged, Condition::GreaterThan));

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
        self.builder.emit(Instruction::Compare(result, left_untagged, right_untagged, Condition::LessThanOrEqual));

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
        self.builder.emit(Instruction::Compare(result, left_untagged, right_untagged, Condition::GreaterThanOrEqual));

        Ok(result)
    }

    fn compile_builtin_println(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        // println takes any number of arguments (including 0)
        // Compile all arguments and ensure they're in registers
        let mut arg_vals = Vec::new();
        for arg in args {
            let val = self.compile(arg)?;
            arg_vals.push(self.builder.assign_new(val));
        }

        let result = self.builder.new_register();
        self.builder.emit(Instruction::Println(result, arg_vals));

        // println returns nil
        Ok(result)
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
        self.builder.emit(Instruction::Compare(result, left, right, Condition::Equal));

        // Compare returns properly tagged boolean (3 or 11)
        Ok(result)
    }

    fn compile_builtin_gc(&mut self, _args: &[Expr]) -> Result<IrValue, String> {
        // __gc takes no arguments and returns nil
        // The CallGC instruction will call trampoline_gc with the current frame pointer
        let result = self.builder.new_register();
        self.builder.emit(Instruction::CallGC(result));
        Ok(result)
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
        self.builder.emit(Instruction::BitAnd(result_untagged, left_untagged, right_untagged));

        let result = self.builder.new_register();
        self.builder.emit(Instruction::Tag(result, result_untagged, IrValue::TaggedConstant(0)));

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
        self.builder.emit(Instruction::BitOr(result_untagged, left_untagged, right_untagged));

        let result = self.builder.new_register();
        self.builder.emit(Instruction::Tag(result, result_untagged, IrValue::TaggedConstant(0)));

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
        self.builder.emit(Instruction::BitXor(result_untagged, left_untagged, right_untagged));

        let result = self.builder.new_register();
        self.builder.emit(Instruction::Tag(result, result_untagged, IrValue::TaggedConstant(0)));

        Ok(result)
    }

    fn compile_builtin_bit_not(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        if args.len() != 1 {
            return Err(format!("bit-not requires 1 argument, got {}", args.len()));
        }

        let operand = self.compile(&args[0])?;

        let operand_untagged = self.builder.new_register();
        self.builder.emit(Instruction::Untag(operand_untagged, operand));

        let result_untagged = self.builder.new_register();
        self.builder.emit(Instruction::BitNot(result_untagged, operand_untagged));

        let result = self.builder.new_register();
        self.builder.emit(Instruction::Tag(result, result_untagged, IrValue::TaggedConstant(0)));

        Ok(result)
    }

    fn compile_builtin_bit_shift_left(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        if args.len() != 2 {
            return Err(format!("bit-shift-left requires 2 arguments, got {}", args.len()));
        }

        let value = self.compile(&args[0])?;
        let amount = self.compile(&args[1])?;

        let value_untagged = self.builder.new_register();
        let amount_untagged = self.builder.new_register();
        self.builder.emit(Instruction::Untag(value_untagged, value));
        self.builder.emit(Instruction::Untag(amount_untagged, amount));

        let result_untagged = self.builder.new_register();
        self.builder.emit(Instruction::BitShiftLeft(result_untagged, value_untagged, amount_untagged));

        let result = self.builder.new_register();
        self.builder.emit(Instruction::Tag(result, result_untagged, IrValue::TaggedConstant(0)));

        Ok(result)
    }

    fn compile_builtin_bit_shift_right(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        if args.len() != 2 {
            return Err(format!("bit-shift-right requires 2 arguments, got {}", args.len()));
        }

        let value = self.compile(&args[0])?;
        let amount = self.compile(&args[1])?;

        let value_untagged = self.builder.new_register();
        let amount_untagged = self.builder.new_register();
        self.builder.emit(Instruction::Untag(value_untagged, value));
        self.builder.emit(Instruction::Untag(amount_untagged, amount));

        let result_untagged = self.builder.new_register();
        self.builder.emit(Instruction::BitShiftRight(result_untagged, value_untagged, amount_untagged));

        let result = self.builder.new_register();
        self.builder.emit(Instruction::Tag(result, result_untagged, IrValue::TaggedConstant(0)));

        Ok(result)
    }

    fn compile_builtin_unsigned_bit_shift_right(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        if args.len() != 2 {
            return Err(format!("unsigned-bit-shift-right requires 2 arguments, got {}", args.len()));
        }

        let value = self.compile(&args[0])?;
        let amount = self.compile(&args[1])?;

        let value_untagged = self.builder.new_register();
        let amount_untagged = self.builder.new_register();
        self.builder.emit(Instruction::Untag(value_untagged, value));
        self.builder.emit(Instruction::Untag(amount_untagged, amount));

        let result_untagged = self.builder.new_register();
        self.builder.emit(Instruction::UnsignedBitShiftRight(result_untagged, value_untagged, amount_untagged));

        let result = self.builder.new_register();
        self.builder.emit(Instruction::Tag(result, result_untagged, IrValue::TaggedConstant(0)));

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
        self.builder.emit(Instruction::LoadConstant(nil_const, IrValue::Null));

        let result = self.builder.new_register();
        self.builder.emit(Instruction::Compare(result, val, nil_const, Condition::Equal));
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
        self.builder.emit(Instruction::LoadConstant(two_const, IrValue::TaggedConstant(2)));

        // Check if tag < 2 (i.e., tag is 0 for int or 1 for float)
        let result = self.builder.new_register();
        self.builder.emit(Instruction::Compare(result, tag, two_const, Condition::LessThan));
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
        self.builder.emit(Instruction::LoadConstant(two_const, IrValue::TaggedConstant(2)));

        let result = self.builder.new_register();
        self.builder.emit(Instruction::Compare(result, tag, two_const, Condition::Equal));
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
        self.builder.emit(Instruction::LoadConstant(four_const, IrValue::TaggedConstant(4)));
        let five_const = self.builder.new_register();
        self.builder.emit(Instruction::LoadConstant(five_const, IrValue::TaggedConstant(5)));

        // Check tag == 4 (function)
        self.builder.emit(Instruction::JumpIf(is_fn_label.clone(), Condition::Equal, tag, four_const));

        // Check tag == 5 (closure)
        self.builder.emit(Instruction::JumpIf(is_fn_label.clone(), Condition::Equal, tag, five_const));

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
            return Err(format!("identical? requires 2 arguments, got {}", args.len()));
        }
        let left = self.compile(&args[0])?;
        let left = self.ensure_register(left);
        let right = self.compile(&args[1])?;
        let right = self.ensure_register(right);

        // Compare raw tagged values - identical values have identical bit patterns
        let result = self.builder.new_register();
        self.builder.emit(Instruction::Compare(result, left, right, Condition::Equal));
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::read;
    use crate::clojure_ast::analyze;
    use std::cell::UnsafeCell;

    #[test]
    fn test_compile_add_generates_ir() {
        let runtime = Arc::new(UnsafeCell::new(GCRuntime::new()));
        let mut compiler = Compiler::new(runtime);
        let val = read("(+ 1 2)").unwrap();
        let ast = analyze(&val).unwrap();

        compiler.compile(&ast).unwrap();
        let instructions = compiler.take_instructions();

        // Polymorphic arithmetic generates many instructions for type dispatch:
        // - LoadConstant for operands
        // - Type tag checks and branching
        // - Int path: Untag, AddInt, Tag
        // - Float path: Type checks, conversions, AddFloat, AllocateFloat
        println!("\nGenerated {} IR instructions for (+ 1 2):", instructions.len());
        for (i, inst) in instructions.iter().enumerate() {
            println!("  {}: {:?}", i, inst);
        }

        // Just verify we generated some instructions (the exact count varies with optimizations)
        assert!(instructions.len() >= 6, "Expected at least 6 instructions, got {}", instructions.len());
    }

    #[test]
    fn test_compile_nested() {
        let runtime = Arc::new(UnsafeCell::new(GCRuntime::new()));
        let mut compiler = Compiler::new(runtime);
        let val = read("(+ (* 2 3) 4)").unwrap();
        let ast = analyze(&val).unwrap();

        compiler.compile(&ast).unwrap();
        let instructions = compiler.take_instructions();

        println!("\nGenerated {} IR instructions for (+ (* 2 3) 4):", instructions.len());
        for (i, inst) in instructions.iter().enumerate() {
            println!("  {}: {:?}", i, inst);
        }

        // Should compile (* 2 3) first, then (+ result 4)
        assert!(instructions.len() > 10);
    }

    #[test]
    fn test_compile_string_literal() {
        let runtime = Arc::new(UnsafeCell::new(GCRuntime::new()));
        let mut compiler = Compiler::new(runtime.clone());
        let val = read("\"hello\"").unwrap();
        let ast = analyze(&val).unwrap();

        compiler.compile(&ast).unwrap();
        let instructions = compiler.take_instructions();

        // Should generate a single LoadConstant instruction
        assert_eq!(instructions.len(), 1);
        match &instructions[0] {
            Instruction::LoadConstant(_, IrValue::TaggedConstant(ptr)) => {
                // Verify it's tagged as a string (tag 0b010)
                assert_eq!(*ptr & 0b111, 0b010, "Should have string tag");
                // Verify we can read it back
                let rt = unsafe { &*runtime.get() };
                let s = rt.read_string(*ptr as usize);
                assert_eq!(s, "hello");
            }
            _ => panic!("Expected LoadConstant instruction"),
        }
    }

    #[test]
    fn test_compile_empty_string() {
        let runtime = Arc::new(UnsafeCell::new(GCRuntime::new()));
        let mut compiler = Compiler::new(runtime.clone());
        let val = read("\"\"").unwrap();
        let ast = analyze(&val).unwrap();

        compiler.compile(&ast).unwrap();
        let instructions = compiler.take_instructions();

        assert_eq!(instructions.len(), 1);
        match &instructions[0] {
            Instruction::LoadConstant(_, IrValue::TaggedConstant(ptr)) => {
                let rt = unsafe { &*runtime.get() };
                let s = rt.read_string(*ptr as usize);
                assert_eq!(s, "");
            }
            _ => panic!("Expected LoadConstant instruction"),
        }
    }

    #[test]
    fn test_compile_string_with_spaces() {
        let runtime = Arc::new(UnsafeCell::new(GCRuntime::new()));
        let mut compiler = Compiler::new(runtime.clone());
        let val = read("\"hello world\"").unwrap();
        let ast = analyze(&val).unwrap();

        compiler.compile(&ast).unwrap();
        let instructions = compiler.take_instructions();

        assert_eq!(instructions.len(), 1);
        match &instructions[0] {
            Instruction::LoadConstant(_, IrValue::TaggedConstant(ptr)) => {
                let rt = unsafe { &*runtime.get() };
                let s = rt.read_string(*ptr as usize);
                assert_eq!(s, "hello world");
            }
            _ => panic!("Expected LoadConstant instruction"),
        }
    }

    #[test]
    fn test_compile_string_in_def() {
        let runtime = Arc::new(UnsafeCell::new(GCRuntime::new()));
        let mut compiler = Compiler::new(runtime.clone());
        let val = read("(def greeting \"hello\")").unwrap();
        let ast = analyze(&val).unwrap();

        // Should compile without error
        compiler.compile(&ast).unwrap();
        let instructions = compiler.take_instructions();

        // Should have instructions for allocating var, storing value, etc.
        assert!(instructions.len() > 1, "Should generate multiple instructions for def");
    }
}
