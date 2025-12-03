# Plan: Implementing Clojure `fn` Special Form

## Overview

This plan implements the `fn` special form following Clojure's specification exactly. The implementation will support all Clojure features including multi-arity functions, variadic arguments, self-recursion, closures, and condition maps.

---

## Clojure `fn` Specification Summary

**Syntax:**
```clojure
(fn name? [params*] expr*)           ; Single arity
(fn name? ([params*] expr*)+)        ; Multi-arity
```

**Parameters:**
```clojure
params ⇒ positional-param* , or positional-param* & rest-param
positional-param ⇒ binding-form
rest-param ⇒ binding-form
name ⇒ symbol
```

**Key Features:**
1. **First-class objects** - Functions are values that can be passed around
2. **Multi-arity** - Single fn object can have multiple invoke methods (arities 0-20)
3. **Variadic** - One arity can collect excess args in rest-param (via `&`)
4. **Self-recursion** - Optional name binds function to itself within body
5. **Closures** - Functions capture lexical environment
6. **Implicit do** - Multiple body expressions wrapped in do
7. **Condition maps** - Optional `:pre` and `:post` assertions (since 1.1)

**Examples:**
```clojure
;; Simple function
(fn [x] (* x x))

;; Named for self-recursion
(fn factorial [n]
  (if (<= n 1)
    1
    (* n (factorial (- n 1)))))

;; Multi-arity
(fn mult
  ([] 1)
  ([x] x)
  ([x y] (* x y))
  ([x y & more]
    (apply mult (mult x y) more)))

;; With condition map
(fn [x]
  {:pre [(pos? x)]
   :post [(> % 16) (< % 225)]}
  (* x x))
```

---

## Phase 1: Core Infrastructure (Days 1-2)

### 1.1 Extend Value Type for Function Objects

**File:** `src/value.rs:32-37`

**Current State:**
```rust
Function {
    name: Option<String>,
    params: Vec<String>,
    // body will be added later
},
```

**Changes Needed:**
Replace the temporary placeholder with a proper function representation:

```rust
Function {
    name: Option<String>,
    arity_map: HashMap<usize, FnArity>,  // arity → implementation
    variadic_arity: Option<Box<FnArity>>,  // Optional variadic overload
},
```

**New struct to add:**
```rust
#[derive(Debug, Clone, PartialEq)]
pub struct FnArity {
    pub params: Vec<String>,           // Positional params
    pub rest_param: Option<String>,    // Variadic rest param (if present)
    pub body: Vec<Expr>,               // Body expressions (implicit do)
    pub pre_conditions: Vec<Expr>,     // :pre assertions
    pub post_conditions: Vec<Expr>,    // :post assertions (can use %)
}
```

**Why this design:**
- Matches Clojure's invoke overloading on arity
- `arity_map` allows O(1) dispatch at call time
- Separate `variadic_arity` ensures only one variadic overload (Clojure constraint)
- Storing `Expr` in body defers compilation until call time (enables closures)

---

### 1.2 Extend AST for `fn` Special Form

**File:** `src/clojure_ast.rs:8-75`

**Add new variant to `Expr` enum:**
```rust
/// (fn name? [params*] exprs*)
/// (fn name? ([params*] exprs*)+)
Fn {
    name: Option<String>,               // Optional self-recursion name
    arities: Vec<FnArity>,              // One or more arity overloads
},
```

**Note:** Reuse the `FnArity` struct from `value.rs` (move to shared location or duplicate for simplicity).

---

### 1.3 Implement Analyzer for `fn`

**File:** `src/clojure_ast.rs:81-158`

**Add to match statement (line 139):**
```rust
"fn" => analyze_fn(items),
```

**Implement `analyze_fn` function:**

Location: Add after `analyze_let` (after line 367)

```rust
fn analyze_fn(items: &im::Vector<Value>) -> Result<Expr, String> {
    // (fn name? ...)
    // Minimum: (fn [params] body)
    if items.len() < 2 {
        return Err("fn requires at least a parameter vector".to_string());
    }

    let mut idx = 1;

    // Check for optional name
    let name = if let Some(Value::Symbol(s)) = items.get(idx) {
        // Could be name or could be param vector
        // Name is a symbol, params is a vector
        // Peek ahead to disambiguate
        if let Some(Value::Vector(_)) = items.get(idx + 1) {
            // This is a name, next is params
            idx += 1;
            Some(s.clone())
        } else if matches!(items.get(idx), Some(Value::Vector(_))) {
            // This is params, no name
            None
        } else if matches!(items.get(idx), Some(Value::List(_))) {
            // Multi-arity form starting with (params) list
            None
        } else {
            return Err("fn requires parameter vector or arity forms".to_string());
        }
    } else {
        None
    };

    // Parse arities
    let mut arities = Vec::new();

    // Check if single-arity or multi-arity form
    let is_multi_arity = matches!(items.get(idx), Some(Value::List(_)));

    if is_multi_arity {
        // Multi-arity: (fn name? ([params] body)+ )
        for i in idx..items.len() {
            let arity_form = match &items[i] {
                Value::List(arity_items) => arity_items,
                _ => return Err("Multi-arity fn requires lists for each arity".to_string()),
            };

            let arity = parse_fn_arity(arity_form)?;
            arities.push(arity);
        }
    } else {
        // Single-arity: (fn name? [params] body*)
        let params_vec = match &items[idx] {
            Value::Vector(v) => v,
            _ => return Err("fn requires parameter vector".to_string()),
        };

        // Collect body expressions
        let body_items = items.slice(idx + 1..);

        let arity = parse_fn_params_and_body(params_vec, &body_items)?;
        arities.push(arity);
    }

    // Validate arities
    validate_fn_arities(&arities)?;

    Ok(Expr::Fn { name, arities })
}

fn parse_fn_arity(arity_items: &im::Vector<Value>) -> Result<FnArity, String> {
    // ([params] condition-map? body*)
    if arity_items.is_empty() {
        return Err("fn arity form cannot be empty".to_string());
    }

    let params_vec = match &arity_items[0] {
        Value::Vector(v) => v,
        _ => return Err("fn arity form must start with parameter vector".to_string()),
    };

    let body_items = arity_items.slice(1..);
    parse_fn_params_and_body(params_vec, &body_items)
}

fn parse_fn_params_and_body(
    params_vec: &im::Vector<Value>,
    body_items: &im::Vector<Value>,
) -> Result<FnArity, String> {
    // Parse parameters: [x y] or [x y & rest]
    let mut params = Vec::new();
    let mut rest_param = None;
    let mut found_ampersand = false;

    for (i, param) in params_vec.iter().enumerate() {
        match param {
            Value::Symbol(s) if s == "&" => {
                if found_ampersand {
                    return Err("Only one & allowed in parameter list".to_string());
                }
                if i == params_vec.len() - 1 {
                    return Err("& must be followed by rest parameter".to_string());
                }
                found_ampersand = true;
            }
            Value::Symbol(s) => {
                if found_ampersand {
                    if rest_param.is_some() {
                        return Err("Only one rest parameter allowed after &".to_string());
                    }
                    rest_param = Some(s.clone());
                } else {
                    params.push(s.clone());
                }
            }
            _ => return Err(format!("fn parameters must be symbols, got {:?}", param)),
        }
    }

    // Parse condition map and body
    let (pre_conditions, post_conditions, body_start_idx) =
        if !body_items.is_empty() {
            if let Some(Value::Map(cond_map)) = body_items.get(0) {
                // Check if this is truly a condition map or just the body
                // Condition map has :pre and/or :post keys
                let has_pre = cond_map.contains_key(&Value::Keyword("pre".to_string()));
                let has_post = cond_map.contains_key(&Value::Keyword("post".to_string()));

                if has_pre || has_post {
                    // Parse condition map
                    let pre = if let Some(Value::Vector(pre_vec)) = cond_map.get(&Value::Keyword("pre".to_string())) {
                        pre_vec.iter()
                            .map(|v| analyze(v))
                            .collect::<Result<Vec<_>, _>>()?
                    } else {
                        Vec::new()
                    };

                    let post = if let Some(Value::Vector(post_vec)) = cond_map.get(&Value::Keyword("post".to_string())) {
                        post_vec.iter()
                            .map(|v| analyze(v))
                            .collect::<Result<Vec<_>, _>>()?
                    } else {
                        Vec::new()
                    };

                    (pre, post, 1)
                } else {
                    // Just a map in the body, not a condition map
                    (Vec::new(), Vec::new(), 0)
                }
            } else {
                (Vec::new(), Vec::new(), 0)
            }
        } else {
            (Vec::new(), Vec::new(), 0)
        };

    // Parse body expressions
    let mut body = Vec::new();
    for i in body_start_idx..body_items.len() {
        body.push(analyze(&body_items[i])?);
    }

    if body.is_empty() {
        // Empty body returns nil
        body.push(Expr::Literal(Value::Nil));
    }

    Ok(FnArity {
        params,
        rest_param,
        body,
        pre_conditions,
        post_conditions,
    })
}

fn validate_fn_arities(arities: &[FnArity]) -> Result<(), String> {
    if arities.is_empty() {
        return Err("fn requires at least one arity".to_string());
    }

    let mut seen_arities = std::collections::HashSet::new();
    let mut variadic_count = 0;

    for arity in arities {
        let arity_num = arity.params.len();

        if arity.rest_param.is_some() {
            variadic_count += 1;
            if variadic_count > 1 {
                return Err("fn can have at most one variadic arity".to_string());
            }
        }

        if !seen_arities.insert(arity_num) {
            return Err(format!("Duplicate arity {} in fn", arity_num));
        }
    }

    Ok(())
}
```

**Complexity:** ~200 lines, handles all parsing edge cases

---

## Phase 2: GC Runtime Support (Day 2)

### 2.1 Add Function Type to GC

**File:** `src/gc_runtime.rs:11-13`

**Add constant:**
```rust
const TYPE_ID_FUNCTION: u8 = 12;
```

### 2.2 Add Function Allocation Method

**File:** `src/gc_runtime.rs` (add to `GCRuntime` impl)

**Add method:**
```rust
/// Allocate a function object on the heap
/// Layout: Header | name_ptr | arity_count | [arity_data...]
///
/// For now, we'll store compiled code pointer and closure data
/// Each arity gets: arity_num | code_ptr | closure_data_ptr
pub fn allocate_function(
    &mut self,
    name: Option<String>,
    compiled_code_ptr: usize,
    closure_vars: Vec<(String, usize)>,  // Captured variables
) -> Result<usize, String> {
    // Calculate size needed
    let name_ptr = if let Some(n) = name {
        self.allocate_string(&n)?
    } else {
        0
    };

    // For Phase 1, simple single-arity functions
    // Size: name_ptr + code_ptr + closure_count + closure_data
    let size_words = 2 + closure_vars.len() * 2;

    let mut obj = self.allocate(TYPE_ID_FUNCTION, size_words, 0, false)?;

    obj.write_field(0, name_ptr);
    obj.write_field(1, compiled_code_ptr);

    // Store closure data
    for (i, (_name, value)) in closure_vars.iter().enumerate() {
        obj.write_field(2 + i * 2, self.allocate_string(_name)?);
        obj.write_field(2 + i * 2 + 1, *value);
    }

    Ok(obj.pointer)
}
```

---

## Phase 3: IR Extensions (Day 3)

### 3.1 Add Call-Related IR Instructions

**File:** `src/ir.rs:56-92`

**Add new instructions:**
```rust
// Function operations
MakeFunction(IrValue, Label),           // MakeFunction(dst, code_label) - create function object
LoadClosure(IrValue, IrValue, usize),   // LoadClosure(dst, fn_obj, index) - load closure var
Call(IrValue, IrValue, Vec<IrValue>),   // Call(dst, fn, args) - invoke function
```

**Explanation:**
- `MakeFunction`: Creates a function object pointing to compiled code
- `LoadClosure`: Reads captured variables from function object
- `Call`: Invokes a function with arguments (checks arity at runtime)

---

### 3.2 Add Function Call Codegen

**File:** `src/arm_codegen.rs` (find the instruction match statement)

**Add cases for new instructions:**
```rust
Instruction::MakeFunction(dst, label) => {
    // Allocate function object on heap
    // Store code pointer (from label)
    // Return tagged heap pointer
    todo!("MakeFunction codegen")
}

Instruction::LoadClosure(dst, fn_obj, index) => {
    // Dereference function object
    // Load closure variable at index
    todo!("LoadClosure codegen")
}

Instruction::Call(dst, fn_val, args) => {
    // Untag function pointer
    // Extract code pointer from function object
    // Setup stack frame
    // Copy arguments to argument registers/stack
    // Branch to function code
    // Store return value
    todo!("Call codegen")
}
```

---

## Phase 4: Compiler Implementation (Days 3-4)

### 4.1 Compile `fn` Expression

**File:** `src/compiler.rs:82-97`

**Update match statement:**
```rust
Expr::Fn { name, arities } => self.compile_fn(name, arities),
```

### 4.2 Implement `compile_fn` Method

**File:** `src/compiler.rs` (add new method)

**Implementation:**
```rust
fn compile_fn(&mut self, name: &Option<String>, arities: &[FnArity]) -> Result<IrValue, String> {
    // Strategy:
    // 1. Detect free variables (variables not in params but referenced in body)
    // 2. Generate code for each arity
    // 3. Generate arity dispatch code
    // 4. Create function object with closures

    // Step 1: Analyze closures
    let free_vars = self.find_free_variables(arities)?;

    // Step 2: Capture free variables (evaluate them in current scope)
    let mut closure_values = Vec::new();
    for var in &free_vars {
        let value = self.compile_var(&None, var)?;
        closure_values.push((var.clone(), value));
    }

    // Step 3: Generate code for each arity
    let mut arity_labels = Vec::new();

    for arity in arities {
        let arity_label = self.builder.new_label();
        self.builder.emit(Instruction::Label(arity_label.clone()));

        // Push new scope with parameters
        self.push_scope();

        // If named, bind function name to itself (for recursion)
        if let Some(fn_name) = name {
            // For now, store function pointer in local scope
            // TODO: This needs the actual function object, not just a placeholder
            // We'll need to handle this specially
        }

        // Bind parameters to argument registers
        for (i, param) in arity.params.iter().enumerate() {
            let arg_reg = IrValue::Register(VirtualRegister {
                index: i,
                is_argument: true,
            });
            self.bind_local(param, arg_reg);
        }

        // Bind rest parameter if variadic
        if let Some(rest) = &arity.rest_param {
            // Collect remaining args into a list
            // For now, just bind to nil
            let rest_reg = self.builder.new_register();
            self.builder.emit(Instruction::LoadConstant(rest_reg, IrValue::Null));
            self.bind_local(rest, rest_reg);
        }

        // Bind closure variables
        for (i, (var_name, _)) in free_vars.iter().enumerate() {
            let closure_reg = self.builder.new_register();
            // TODO: Load from function object closure data
            self.bind_local(var_name, closure_reg);
        }

        // Evaluate pre-conditions (if *assert* is true)
        for pre in &arity.pre_conditions {
            let result = self.compile(pre)?;
            // Check if false, throw AssertionError
            // For now, skip (assertions not implemented yet)
        }

        // Compile body (implicit do)
        let result = self.compile_body(&arity.body)?;

        // Evaluate post-conditions (if *assert* is true)
        for post in &arity.post_conditions {
            // Special: % refers to return value
            // Need to handle this specially
            // For now, skip
        }

        // Return result
        self.builder.emit(Instruction::Ret(result));

        self.pop_scope();

        arity_labels.push((arity.params.len(), arity_label));
    }

    // Step 4: Generate dispatch code
    let dispatch_label = self.builder.new_label();
    self.builder.emit(Instruction::Label(dispatch_label.clone()));

    // Dispatch logic:
    // - Check argument count
    // - Jump to appropriate arity
    // - Fall through to variadic if count exceeds all fixed arities
    // For now, just assume single arity (multi-arity in Phase 5)

    if arity_labels.len() == 1 {
        let (expected_arity, arity_label) = &arity_labels[0];
        // For single arity, no dispatch needed - just jump
        self.builder.emit(Instruction::Jump(arity_label.clone()));
    } else {
        // Multi-arity dispatch (implement in Phase 5)
        return Err("Multi-arity functions not yet implemented".to_string());
    }

    // Step 5: Create function object
    let fn_obj = self.builder.new_register();
    self.builder.emit(Instruction::MakeFunction(fn_obj, dispatch_label));

    // TODO: Store closure values in function object

    Ok(fn_obj)
}

fn compile_body(&mut self, exprs: &[Expr]) -> Result<IrValue, String> {
    let mut result = self.builder.new_register();
    self.builder.emit(Instruction::LoadConstant(result, IrValue::Null));

    for expr in exprs {
        result = self.compile(expr)?;
    }

    Ok(result)
}

fn find_free_variables(&self, arities: &[FnArity]) -> Result<Vec<String>, String> {
    // Scan all arity bodies for variable references
    // A variable is free if:
    // 1. It's referenced in the body
    // 2. It's not a parameter
    // 3. It's not a global var
    // 4. It's not bound by an inner let

    let mut free_vars = std::collections::HashSet::new();

    for arity in arities {
        let bound_vars: std::collections::HashSet<_> =
            arity.params.iter()
                .chain(arity.rest_param.iter())
                .cloned()
                .collect();

        self.collect_free_vars_from_exprs(&arity.body, &bound_vars, &mut free_vars);
    }

    Ok(free_vars.into_iter().collect())
}

fn collect_free_vars_from_exprs(
    &self,
    exprs: &[Expr],
    bound: &std::collections::HashSet<String>,
    free: &mut std::collections::HashSet<String>,
) {
    for expr in exprs {
        self.collect_free_vars_from_expr(expr, bound, free);
    }
}

fn collect_free_vars_from_expr(
    &self,
    expr: &Expr,
    bound: &std::collections::HashSet<String>,
    free: &mut std::collections::HashSet<String>,
) {
    match expr {
        Expr::Var { namespace: None, name } => {
            // Check if it's bound locally or in outer scope
            if !bound.contains(name) && !self.is_builtin(name) {
                // Check if it's a local in current scope
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
            self.collect_free_vars_from_exprs(body, &new_bound, free);
        }
        Expr::Fn { name, arities } => {
            // Nested function - need to recurse but with proper scoping
            // Name (if present) is bound within the function
            let mut fn_bound = bound.clone();
            if let Some(n) = name {
                fn_bound.insert(n.clone());
            }

            for arity in arities {
                let mut arity_bound = fn_bound.clone();
                for param in &arity.params {
                    arity_bound.insert(param.clone());
                }
                if let Some(rest) = &arity.rest_param {
                    arity_bound.insert(rest.clone());
                }
                self.collect_free_vars_from_exprs(&arity.body, &arity_bound, free);
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
            self.collect_free_vars_from_exprs(exprs, bound, free);
        }
        Expr::Call { func, args } => {
            self.collect_free_vars_from_expr(func, bound, free);
            self.collect_free_vars_from_exprs(args, bound, free);
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
            self.collect_free_vars_from_exprs(body, bound, free);
        }
        _ => {
            // Literals, quotes, ns, use - no free variables
        }
    }
}
```

**Complexity:** ~250 lines, core of the implementation

---

### 4.3 Update Function Call Compilation

**File:** `src/compiler.rs:568-757` (in `compile_call`)

**Current state:** Only handles built-ins

**Changes needed:**
```rust
fn compile_call(&mut self, func_expr: &Expr, args: &[Expr]) -> Result<IrValue, String> {
    // Check if func is a built-in first
    if let Expr::Var { namespace: None, name } = func_expr {
        if self.is_builtin(name) {
            return self.compile_builtin_call(name, args);
        }
    }

    // General function call
    // 1. Evaluate function expression
    let fn_val = self.compile(func_expr)?;

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

fn compile_builtin_call(&mut self, name: &str, args: &[Expr]) -> Result<IrValue, String> {
    // Move existing built-in dispatch logic here
    // This is the current compile_call implementation
    // ... (existing code for +, -, *, /, <, >, =)
}
```

---

## Phase 5: Multi-Arity Dispatch (Day 4)

### 5.1 Implement Arity Dispatch Logic

**Location:** In `compile_fn` method, replace dispatch placeholder

**Implementation:**
```rust
// Generate dispatch code for multi-arity functions
let arg_count_reg = self.builder.new_register();
// TODO: Get argument count at runtime
// For now, this requires passing arg count to function calls

// Sort arities by param count
let mut sorted_arities = arity_labels.clone();
sorted_arities.sort_by_key(|(arity, _)| *arity);

// Generate cascade of comparisons
for (i, (arity, label)) in sorted_arities.iter().enumerate() {
    if i == sorted_arities.len() - 1 {
        // Last arity - could be variadic
        // Check if variadic (has rest param)
        let is_variadic = arities.iter()
            .find(|a| a.params.len() == *arity)
            .and_then(|a| a.rest_param.as_ref())
            .is_some();

        if is_variadic {
            // Jump if arg_count >= arity
            let else_label = self.builder.new_label();
            self.builder.emit(Instruction::JumpIf(
                else_label.clone(),
                Condition::LessThan,
                arg_count_reg,
                IrValue::TaggedConstant(*arity as isize),
            ));
            self.builder.emit(Instruction::Jump(label.clone()));
            self.builder.emit(Instruction::Label(else_label));

            // Fall through to arity error
        } else {
            // Jump if arg_count == arity
            let else_label = self.builder.new_label();
            self.builder.emit(Instruction::JumpIf(
                else_label.clone(),
                Condition::NotEqual,
                arg_count_reg,
                IrValue::TaggedConstant(*arity as isize),
            ));
            self.builder.emit(Instruction::Jump(label.clone()));
            self.builder.emit(Instruction::Label(else_label));
        }
    } else {
        // Jump if arg_count == arity
        let next_label = self.builder.new_label();
        self.builder.emit(Instruction::JumpIf(
            next_label.clone(),
            Condition::NotEqual,
            arg_count_reg,
            IrValue::TaggedConstant(*arity as isize),
        ));
        self.builder.emit(Instruction::Jump(label.clone()));
        self.builder.emit(Instruction::Label(next_label));
    }
}

// Arity error - throw exception
// For now, return nil
let error_reg = self.builder.new_register();
self.builder.emit(Instruction::LoadConstant(error_reg, IrValue::Null));
self.builder.emit(Instruction::Ret(error_reg));
```

---

## Phase 6: Testing (Day 5)

### 6.1 Basic Function Tests

**File:** `tests/test_fn.txt` (create new file)

```clojure
;; Test 1: Simple function
(def square (fn [x] (* x x)))
(square 5)  ;; => 25

;; Test 2: Named function for recursion
(def factorial
  (fn fact [n]
    (if (<= n 1)
      1
      (* n (fact (- n 1))))))
(factorial 5)  ;; => 120

;; Test 3: Closure
(def make-adder
  (fn [x]
    (fn [y] (+ x y))))
(def add5 (make-adder 5))
(add5 10)  ;; => 15

;; Test 4: Multi-arity
(def mult
  (fn
    ([] 1)
    ([x] x)
    ([x y] (* x y))))
(mult)      ;; => 1
(mult 5)    ;; => 5
(mult 3 4)  ;; => 12

;; Test 5: Variadic
(def sum
  (fn [x & more]
    ;; For now, just return x (list operations not implemented)
    x))
(sum 1)      ;; => 1
(sum 1 2 3)  ;; => 1 (until we implement list ops)
```

### 6.2 Condition Map Tests (Optional)

**File:** `tests/test_fn_conditions.txt`

```clojure
;; Test with pre-conditions
(def safe-div
  (fn [x y]
    {:pre [(> y 0)]}
    (/ x y)))

(safe-div 10 2)  ;; => 5
;; (safe-div 10 0)  ;; => AssertionError (when *assert* is true)

;; Test with post-conditions
(def constrained-sqr
  (fn [x]
    {:pre [(> x 4)]
     :post [(> % 16) (< % 225)]}
    (* x x)))

(constrained-sqr 5)  ;; => 25
;; (constrained-sqr 3)  ;; => AssertionError (pre-condition)
;; (constrained-sqr 20) ;; => AssertionError (post-condition)
```

---

## Phase 7: Edge Cases & Polish (Day 5)

### 7.1 Handle Special Cases

**Cases to test:**
1. **Empty body** - Should return nil
   ```clojure
   (fn [])  ;; => #<fn>, calling it returns nil
   ```

2. **Zero arity**
   ```clojure
   (fn [] 42)  ;; => #<fn>, calling it returns 42
   ```

3. **Condition map as body** (NOT a condition map)
   ```clojure
   (fn [x] {:foo x})  ;; Returns map, not condition map
   ```

4. **Nested functions**
   ```clojure
   (fn [x] (fn [y] (+ x y)))  ;; Closure within closure
   ```

5. **Self-recursion with same name as outer var**
   ```clojure
   (def f (fn f [n] (if (< n 0) 0 (f (- n 1)))))
   ```

---

## Known Limitations (Future Work)

### Not Implemented in This Phase:

1. **Variadic argument collection** - `rest-param` will be bound to nil, not a seq
   - Requires implementing persistent list construction at runtime
   - Needs `apply` function to work properly

2. **Pre/Post-conditions** - Parsed but not enforced
   - Requires `*assert*` dynamic var support
   - Needs exception throwing mechanism

3. **Destructuring in params** - Only simple symbols supported
   - `(fn [[x y]] ...)` not supported yet
   - `(fn [{:keys [a b]}] ...)` not supported yet

4. **Type hints/metadata on params** - Parsed but ignored
   - `(fn [^String x] ...)` accepted but no type checking

5. **Recur optimization** - Can use named recursion, but not TCO
   - `recur` special form not implemented yet
   - No tail-call optimization

6. **defn macro** - Must use `(def name (fn ...))` form
   - `defn` is syntactic sugar, implement later

---

## Dependencies & Prerequisites

### Must be working:
- ✅ `let` bindings (for closure capture and parameters)
- ✅ `if` conditionals (for examples)
- ✅ Arithmetic operators `+`, `-`, `*`, `/`
- ✅ Comparison operators `<`, `>`, `=`
- ✅ GC heap allocation
- ✅ ARM64 codegen for existing instructions

### Nice to have (for full testing):
- ❌ `apply` function (for variadic functions)
- ❌ `throw` and `catch` (for condition maps)
- ❌ List construction at runtime

---

## Risk Assessment

### High Risk:
1. **Closure capture** - Complex interaction with scope management
   - Mitigation: Start with non-closure functions, add closures incrementally

2. **Multi-arity dispatch** - Runtime complexity
   - Mitigation: Implement single-arity first, add multi-arity as enhancement

3. **ARM64 calling convention** - Getting stack frames right
   - Mitigation: Study existing Beagle implementation carefully

### Medium Risk:
1. **Function object GC** - Need to trace closures correctly
   - Mitigation: Mark function objects as containing pointers

2. **Self-recursion binding** - Chicken-and-egg problem (need fn object to bind name)
   - Mitigation: Use placeholder or late binding

### Low Risk:
1. **Parsing** - Well-specified syntax
2. **AST representation** - Straightforward mapping

---

## Success Criteria

### Phase 1-3 (MVP):
- ✅ Parse `(fn [x] body)` syntax correctly
- ✅ Compile simple functions without closures
- ✅ Call functions with correct arity
- ✅ Return values work correctly

### Phase 4 (Closures):
- ✅ Capture free variables from outer scope
- ✅ Access captured variables in function body
- ✅ Nested closures work

### Phase 5 (Multi-arity):
- ✅ Define functions with multiple arities
- ✅ Dispatch to correct arity at call time
- ✅ Error on wrong arity

### Phase 6 (Complete):
- ✅ Self-recursion with named functions
- ✅ All basic tests pass
- ✅ Integration with existing special forms

---

## Timeline Estimate

- **Day 1:** Phase 1 (AST, Value types, Analyzer) - 6 hours
- **Day 2:** Phase 2 (GC support) + Phase 3 (IR) - 8 hours
- **Day 3:** Phase 4 (Compiler, single-arity) - 8 hours
- **Day 4:** Phase 5 (Multi-arity) + Phase 6 (Testing) - 8 hours
- **Day 5:** Phase 7 (Edge cases, polish) - 4 hours

**Total:** 34 hours (~4-5 days)

This is an aggressive but achievable timeline for someone familiar with the codebase.

---

## Next Steps

1. **Review this plan** - Get approval on approach
2. **Ask clarifying questions** - Resolve any ambiguities
3. **Start Phase 1** - Begin with AST changes
4. **Iterate with tests** - Write tests alongside implementation
5. **Document as we go** - Keep this plan updated with findings

---

## Questions for User

Before proceeding, please clarify:

1. **Variadic functions** - Should we implement rest parameter collection in Phase 1, or defer until list operations are ready?

2. **Condition maps** - Should we implement assertion checking, or just parse and ignore for now?

3. **Multi-arity** - Should we tackle this in Phase 1, or focus on single-arity first?

4. **Destructuring** - Out of scope for this phase?

5. **Performance** - Any specific performance targets for function calls?

---

**End of Plan**
