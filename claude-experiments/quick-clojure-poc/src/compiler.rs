use crate::clojure_ast::Expr;
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

    /// Local variable scopes (for let bindings)
    /// Each scope maps variable name → register
    /// Stack of scopes allows for nested lets
    local_scopes: Vec<HashMap<String, IrValue>>,

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

    /// Type registry for deftype: type_name → (type_id, field_names)
    type_registry: HashMap<String, (usize, Vec<String>)>,
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
            let ns_var_ptr = rt.allocate_var(core_ns_ptr, "*ns*", user_ns_ptr).unwrap();
            let new_core_ns_ptr = rt.namespace_add_binding(core_ns_ptr, "*ns*", ns_var_ptr).unwrap();

            // Mark *ns* as dynamic so it can be rebound later
            rt.mark_var_dynamic(ns_var_ptr);

            // Update core_ns_ptr if it changed (due to namespace re-allocation)
            let core_ns_ptr = new_core_ns_ptr;

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
        }
    }

    fn compile_literal(&mut self, value: &Value) -> Result<IrValue, String> {
        let result = self.builder.new_register();

        match value {
            Value::Nil => {
                self.builder.emit(Instruction::LoadConstant(result, IrValue::Null));
            }
            Value::Bool(true) => {
                self.builder.emit(Instruction::LoadTrue(result));
            }
            Value::Bool(false) => {
                self.builder.emit(Instruction::LoadFalse(result));
            }
            Value::Int(i) => {
                // For now, just store the raw value - we'll add tagging later
                let tagged = (*i as isize) << 3;  // Simple 3-bit tag (000 for int)
                self.builder.emit(Instruction::LoadConstant(
                    result,
                    IrValue::TaggedConstant(tagged),
                ));
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
                let result = self.builder.new_register();
                // Check if var is dynamic at compile time
                if rt.is_var_dynamic(var_ptr) {
                    // Dynamic var - use trampoline for binding lookup
                    self.builder.emit(Instruction::LoadVarDynamic(
                        result,
                        IrValue::TaggedConstant(var_ptr as isize),
                    ));
                } else {
                    // Non-dynamic var - direct memory load
                    self.builder.emit(Instruction::LoadVar(
                        result,
                        IrValue::TaggedConstant(var_ptr as isize),
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
                            let result = self.builder.new_register();
                            // Check if var is dynamic at compile time
                            if rt.is_var_dynamic(var_ptr) {
                                self.builder.emit(Instruction::LoadVarDynamic(
                                    result,
                                    IrValue::TaggedConstant(var_ptr as isize),
                                ));
                            } else {
                                self.builder.emit(Instruction::LoadVar(
                                    result,
                                    IrValue::TaggedConstant(var_ptr as isize),
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
            let result = self.builder.new_register();
            // Check if var is dynamic at compile time
            if rt.is_var_dynamic(var_ptr) {
                // Dynamic var - use trampoline for binding lookup
                self.builder.emit(Instruction::LoadVarDynamic(
                    result,
                    IrValue::TaggedConstant(var_ptr as isize),
                ));
            } else {
                // Non-dynamic var - direct memory load
                self.builder.emit(Instruction::LoadVar(
                    result,
                    IrValue::TaggedConstant(var_ptr as isize),
                ));
            }
            Ok(result)
        } else {
            Err(format!("Undefined variable: {}/{}", namespace.as_ref().unwrap(), name))
        }
    }

    /// Check if a symbol is a built-in function
    fn is_builtin(&self, name: &str) -> bool {
        matches!(name, "+" | "-" | "*" | "/" | "<" | ">" | "=")
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

    /// Compile (deftype* TypeName [field1 field2 ...])
    /// Registers the type and returns nil
    fn compile_deftype(&mut self, name: &str, fields: &[String]) -> Result<IrValue, String> {
        // Register the type with the runtime using fully qualified name
        let current_ns = self.get_current_namespace();
        let qualified_name = format!("{}/{}", current_ns, name);

        // Get mutable access to runtime to register type
        let type_id = {
            let rt = unsafe { &mut *self.runtime.get() };
            rt.register_type(qualified_name.clone(), fields.to_vec())
        };

        // Store in compiler's type registry with FULLY QUALIFIED name
        // This prevents namespace collisions and enables qualified constructor calls
        self.type_registry.insert(qualified_name, (type_id, fields.to_vec()));

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
        let (type_id, fields) = self.type_registry.get(&qualified_name)
            .cloned()
            .ok_or_else(|| format!("Unknown type: {}", qualified_name))?;

        // Check arg count matches field count
        if args.len() != fields.len() {
            return Err(format!(
                "Type {} requires {} arguments, got {}",
                type_name, fields.len(), args.len()
            ));
        }

        // Compile all arguments
        let mut arg_values = Vec::new();
        for arg in args {
            arg_values.push(self.compile(arg)?);
        }

        // Emit MakeType instruction
        let result = self.builder.new_register();
        self.builder.emit(Instruction::MakeType(result, type_id, arg_values));

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
        let var_ptr = unsafe {
            let rt = &mut *self.runtime.get();

            // Check if var already exists
            if let Some(existing_var_ptr) = rt.namespace_lookup(self.current_namespace_ptr, name) {
                existing_var_ptr
            } else {
                // Create new var with nil placeholder (7) - will be overwritten after compilation
                let new_var_ptr = rt.allocate_var(
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
                if new_ns_ptr != self.current_namespace_ptr {
                    self.current_namespace_ptr = new_ns_ptr;

                    // Update in registry
                    let ns_name = rt.namespace_name(new_ns_ptr);
                    self.namespace_registry.insert(ns_name.clone(), new_ns_ptr);

                    // Update GC root
                    rt.add_namespace_root(ns_name, new_ns_ptr);
                }

                new_var_ptr
            }
        };

        // NOW compile the value expression - recursive refs will resolve via LoadVar
        let value_reg = self.compile(value_expr)?;

        // Emit StoreVar instruction to update the var at runtime
        // Pass var_ptr as a tagged constant directly
        self.builder.emit(Instruction::StoreVar(
            IrValue::TaggedConstant(var_ptr as isize),
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
        // Look up the *ns* var pointer
        let ns_var_ptr = unsafe {
            let rt = &*self.runtime.get();
            let core_ns_ptr = *self.namespace_registry.get("clojure.core").unwrap();
            rt.namespace_lookup(core_ns_ptr, "*ns*").unwrap()
        };

        // Load the current namespace pointer as a tagged constant
        let ns_value = self.builder.new_register();
        self.builder.emit(Instruction::LoadConstant(
            ns_value,
            IrValue::TaggedConstant((self.current_namespace_ptr << 3) as isize)
        ));

        // Store it into *ns* var
        self.builder.emit(Instruction::StoreVar(
            IrValue::TaggedConstant(ns_var_ptr as isize),
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
        let mut binding_registers = Vec::new();
        for (name, value_expr) in bindings {
            let value_reg = self.compile(value_expr)?;
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

        // Then assign to binding registers
        for (binding_reg, new_value) in context.binding_registers.iter().zip(new_values.iter()) {
            if binding_reg != new_value {
                self.builder.emit(Instruction::Assign(*binding_reg, *new_value));
            }
        }

        // Jump back to loop
        self.builder.emit(Instruction::Jump(context.label));

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
            scope.insert(name, register);
        }
    }

    /// Look up a local variable, searching from innermost to outermost scope
    fn lookup_local(&self, name: &str) -> Option<IrValue> {
        // Search from innermost scope outward
        for scope in self.local_scopes.iter().rev() {
            if let Some(register) = scope.get(name) {
                return Some(*register);
            }
        }
        None
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
        // Each function is compiled separately with its own register allocation

        if arities.len() != 1 {
            return Err("Multi-arity functions not yet implemented".to_string());
        }

        let arity = &arities[0];

        // Step 1: Analyze closures - find free variables
        let free_vars = self.find_free_variables_in_arity(arity, name)?;

        // Step 2: Capture free variables (evaluate them in current scope)
        // IMPORTANT: This must happen BEFORE compiling the function body to capture
        // the correct registers from the outer scope
        let mut closure_values = Vec::new();
        for var_name in &free_vars {
            if let Some(value_reg) = self.lookup_local(var_name) {
                closure_values.push(value_reg);
            } else {
                return Err(format!("Free variable not found in scope: {}", var_name));
            }
        }

        // Step 3: Save the current IR builder and create a new one for the function
        let outer_builder = std::mem::replace(&mut self.builder, IrBuilder::new());

        // Push new scope for function parameters
        self.push_scope();

        // Create a virtual register for the function object (self) if this function has closures
        // For closures, the closure object is passed as the first argument (x0)
        let self_reg = if !free_vars.is_empty() {
            Some(IrValue::Register(crate::ir::VirtualRegister::Argument(0)))
        } else {
            None
        };

        // Bind parameters to argument registers
        // ARM64 calling convention:
        // - Regular functions: arguments are in x0, x1, x2, etc.
        // - Closures: x0 = closure object, user args in x1, x2, x3, etc.
        let param_offset = if self_reg.is_some() { 1 } else { 0 };
        let mut param_registers = Vec::new();
        for (i, param) in arity.params.iter().enumerate() {
            let arg_reg = IrValue::Register(crate::ir::VirtualRegister::Argument(i + param_offset));
            self.bind_local(param.clone(), arg_reg);
            param_registers.push(arg_reg);
        }

        // Bind rest parameter if variadic
        if let Some(rest) = &arity.rest_param {
            let rest_reg = self.builder.new_register();
            self.builder.emit(Instruction::LoadConstant(rest_reg, IrValue::Null));
            self.bind_local(rest.clone(), rest_reg);
        }

        // Bind closure variables
        // Load them from the function object (self_reg = x0 for closures) using LoadClosure
        if let Some(self_reg) = &self_reg {
            for (i, var_name) in free_vars.iter().enumerate() {
                let closure_reg = self.builder.new_register();
                self.builder.emit(Instruction::LoadClosure(closure_reg, self_reg.clone(), i));
                self.bind_local(var_name.clone(), closure_reg);
            }
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

        // Return result
        self.builder.emit(Instruction::Ret(result));

        // Pop function scope
        self.pop_scope();

        // Step 4: Compile this function's IR separately
        let fn_instructions = self.builder.take_instructions();

        // Store the function IR for display purposes (before compiling)
        self.compiled_function_irs.push((name.clone(), fn_instructions.clone()));

        let code_ptr = Arm64CodeGen::compile_function(&fn_instructions)?;

        // Step 5: Restore the outer IR builder
        self.builder = outer_builder;

        // Step 6: Create function object with closure values and the compiled code pointer
        let fn_obj_reg = self.builder.new_register();
        self.builder.emit(Instruction::MakeFunctionPtr(fn_obj_reg, code_ptr, closure_values));

        Ok(fn_obj_reg)
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
                    if self.lookup_local(name).is_some() {
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
        }
    }

    fn compile_call(&mut self, func: &Expr, args: &[Expr]) -> Result<IrValue, String> {
        // Check if it's a built-in first
        if let Expr::Var { namespace, name } = func {
            // Check if it's a built-in (either explicitly qualified or unqualified)
            let is_builtin = match namespace {
                Some(ns) if ns == "clojure.core" => self.is_builtin(name),
                None => self.is_builtin(name),
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
                    "=" => self.compile_builtin_eq(args),
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

        // 3. Emit Call instruction
        let result = self.builder.new_register();
        self.builder.emit(Instruction::Call(result, fn_val, arg_vals));

        Ok(result)
    }

    fn compile_builtin_add(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        if args.len() != 2 {
            return Err(format!("+ requires 2 arguments, got {}", args.len()));
        }

        let left = self.compile(&args[0])?;
        let right = self.compile(&args[1])?;

        // Untag inputs
        let left_untagged = self.builder.new_register();
        let right_untagged = self.builder.new_register();
        self.builder.emit(Instruction::Untag(left_untagged, left));
        self.builder.emit(Instruction::Untag(right_untagged, right));

        // Add
        let sum = self.builder.new_register();
        self.builder.emit(Instruction::AddInt(sum, left_untagged, right_untagged));

        // Tag result (shift left 3 for int tag 000)
        let result = self.builder.new_register();
        let tag = IrValue::TaggedConstant(0);  // Int tag is 000
        self.builder.emit(Instruction::Tag(result, sum, tag));

        Ok(result)
    }

    fn compile_builtin_sub(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        if args.len() != 2 {
            return Err(format!("- requires 2 arguments, got {}", args.len()));
        }

        let left = self.compile(&args[0])?;
        let right = self.compile(&args[1])?;

        let left_untagged = self.builder.new_register();
        let right_untagged = self.builder.new_register();
        self.builder.emit(Instruction::Untag(left_untagged, left));
        self.builder.emit(Instruction::Untag(right_untagged, right));

        let diff = self.builder.new_register();
        self.builder.emit(Instruction::Sub(diff, left_untagged, right_untagged));

        let result = self.builder.new_register();
        let tag = IrValue::TaggedConstant(0);
        self.builder.emit(Instruction::Tag(result, diff, tag));

        Ok(result)
    }

    fn compile_builtin_mul(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        if args.len() != 2 {
            return Err(format!("* requires 2 arguments, got {}", args.len()));
        }

        let left = self.compile(&args[0])?;
        let right = self.compile(&args[1])?;

        let left_untagged = self.builder.new_register();
        let right_untagged = self.builder.new_register();
        self.builder.emit(Instruction::Untag(left_untagged, left));
        self.builder.emit(Instruction::Untag(right_untagged, right));

        let product = self.builder.new_register();
        self.builder.emit(Instruction::Mul(product, left_untagged, right_untagged));

        let result = self.builder.new_register();
        let tag = IrValue::TaggedConstant(0);
        self.builder.emit(Instruction::Tag(result, product, tag));

        Ok(result)
    }

    fn compile_builtin_div(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        if args.len() != 2 {
            return Err(format!("/ requires 2 arguments, got {}", args.len()));
        }

        let left = self.compile(&args[0])?;
        let right = self.compile(&args[1])?;

        let left_untagged = self.builder.new_register();
        let right_untagged = self.builder.new_register();
        self.builder.emit(Instruction::Untag(left_untagged, left));
        self.builder.emit(Instruction::Untag(right_untagged, right));

        let quotient = self.builder.new_register();
        self.builder.emit(Instruction::Div(quotient, left_untagged, right_untagged));

        let result = self.builder.new_register();
        let tag = IrValue::TaggedConstant(0);
        self.builder.emit(Instruction::Tag(result, quotient, tag));

        Ok(result)
    }

    fn compile_builtin_lt(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        if args.len() != 2 {
            return Err(format!("< requires 2 arguments, got {}", args.len()));
        }

        let left = self.compile(&args[0])?;
        let right = self.compile(&args[1])?;

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

        let left_untagged = self.builder.new_register();
        let right_untagged = self.builder.new_register();
        self.builder.emit(Instruction::Untag(left_untagged, left));
        self.builder.emit(Instruction::Untag(right_untagged, right));

        let result = self.builder.new_register();
        self.builder.emit(Instruction::Compare(result, left_untagged, right_untagged, Condition::GreaterThan));

        // Compare now returns properly tagged boolean (3 or 11)
        Ok(result)
    }

    fn compile_builtin_eq(&mut self, args: &[Expr]) -> Result<IrValue, String> {
        if args.len() != 2 {
            return Err(format!("= requires 2 arguments, got {}", args.len()));
        }

        let left = self.compile(&args[0])?;
        let right = self.compile(&args[1])?;

        // Compare tagged values directly - this preserves type information
        // nil (7), false (3), true (11), and integers (n<<3) all have different representations
        let result = self.builder.new_register();
        self.builder.emit(Instruction::Compare(result, left, right, Condition::Equal));

        // Compare returns properly tagged boolean (3 or 11)
        Ok(result)
    }

    /// Get the generated IR instructions without consuming the compiler
    /// This clears the instruction buffer, allowing the compiler to be reused
    pub fn take_instructions(&mut self) -> Vec<Instruction> {
        self.builder.take_instructions()
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

        // Should generate:
        // 1. LoadConstant for 1
        // 2. LoadConstant for 2
        // 3. Untag left
        // 4. Untag right
        // 5. AddInt
        // 6. Tag result
        println!("\nGenerated {} IR instructions for (+ 1 2):", instructions.len());
        for (i, inst) in instructions.iter().enumerate() {
            println!("  {}: {:?}", i, inst);
        }

        assert_eq!(instructions.len(), 6);
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
}
