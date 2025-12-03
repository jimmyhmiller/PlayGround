# Clojure Special Forms Implementation Roadmap

## Current Status

### ✅ Already Implemented (8/18)
1. ✅ **def** - Define vars with metadata support
2. ✅ **if** - Conditional expressions
3. ✅ **do** - Sequential expression evaluation
4. ✅ **quote** - Prevent evaluation
5. ✅ **set!** - Assignment (for dynamic bindings)
6. ✅ **ns** - Namespace declaration (custom, not standard Clojure)
7. ✅ **use** - Import namespaces (custom, not standard Clojure)
8. ✅ **binding** - Dynamic binding (custom, not standard Clojure)

### ❌ Missing Core Special Forms (10/18)
1. ❌ **let** - Local bindings ⭐ **HIGHEST PRIORITY**
2. ❌ **fn** - Function definitions ⭐ **HIGHEST PRIORITY**
3. ❌ **loop** - Recursion point
4. ❌ **recur** - Tail recursion
5. ❌ **var** - Get var object
6. ❌ **throw** - Throw exceptions
7. ❌ **try/catch/finally** - Exception handling
8. ❌ **monitor-enter/monitor-exit** - Low-level synchronization
9. ❌ **.** (dot) - Java interop (method/field access)
10. ❌ **new** - Java object creation

## Priority Ranking

### Tier 1: Essential (Required for Basic Clojure)
These are **absolutely critical** - without them, you can't write meaningful Clojure code:

1. **let** - Local bindings
   - **Why critical**: Can't write any non-trivial function without local variables
   - **Example**: `(let [x 10 y 20] (+ x y))`
   - **Complexity**: Medium
   - **Estimated time**: 1-2 days

2. **fn** - Function definitions
   - **Why critical**: Can't define custom functions
   - **Example**: `(fn [x] (* x x))`
   - **Complexity**: High (needs closures, multiple arities)
   - **Estimated time**: 3-5 days

### Tier 2: Important (Needed for Real Programs)
These make programs actually useful:

3. **loop/recur** - Iteration and recursion
   - **Why important**: No iteration without these (no `for`, `while`, etc.)
   - **Example**: `(loop [i 0] (if (< i 10) (recur (inc i)) i))`
   - **Complexity**: High (tail-call optimization)
   - **Estimated time**: 2-3 days

4. **try/catch/finally** - Error handling
   - **Why important**: Can't handle errors gracefully
   - **Example**: `(try (/ 1 0) (catch Exception e "error"))`
   - **Complexity**: Medium-High (needs exception handling in JIT)
   - **Estimated time**: 2-3 days

### Tier 3: Nice to Have (Advanced Features)
These are for advanced use cases:

5. **var** - Var introspection
   - **Why useful**: Metaprogramming, REPL tools
   - **Example**: `(var *x*)` returns the Var object
   - **Complexity**: Low
   - **Estimated time**: 1 day

6. **throw** - Exception throwing
   - **Why useful**: Custom error handling
   - **Example**: `(throw (Exception. "error"))`
   - **Complexity**: Medium
   - **Estimated time**: 1 day

### Tier 4: Skip for Now (Low Priority)
These are rarely used or require Java interop:

7. **monitor-enter/monitor-exit** - Low-level locks
   - **Why skip**: Almost never used directly (use `locking` macro)
   - **Complexity**: Medium

8. **.** (dot) - Java interop
   - **Why skip**: Requires Java FFI
   - **Example**: `(.length "hello")`
   - **Complexity**: Very High

9. **new** - Java object creation
   - **Why skip**: Requires Java FFI
   - **Example**: `(new String "hello")`
   - **Complexity**: Very High

## Recommended Implementation Order

### Phase 1: Essential Foundation (Week 1-2)
**Goal**: Make the language actually usable

#### Step 1: Implement `let` (1-2 days) ⭐
**Why first**: Needed by almost everything else

**What to implement**:
```clojure
(let [x 10
      y 20
      z (+ x y)]
  (* z 2))  ;=> 60
```

**Requirements**:
- Sequential binding evaluation
- Lexical scoping
- Shadow outer bindings
- Support destructuring (later)

**AST**:
```rust
Expr::Let {
    bindings: Vec<(String, Box<Expr>)>,
    body: Vec<Box<Expr>>,
}
```

**Compilation strategy**:
- Allocate stack slots for each binding
- Compile each binding value
- Store in stack slot
- Compile body with extended environment
- Clean up stack

#### Step 2: Implement `fn` (3-5 days) ⭐
**Why second**: Enables custom functions

**What to implement**:
```clojure
;; Simple function
(fn [x] (* x x))

;; Multiple arity
(fn
  ([x] (* x x))
  ([x y] (+ x y)))

;; With name (for recursion)
(fn factorial [n]
  (if (<= n 1)
    1
    (* n (factorial (- n 1)))))
```

**Requirements**:
- Closure capture
- Multiple arities
- Recursive self-reference
- Variadic arguments (later)

**AST**:
```rust
Expr::Fn {
    name: Option<String>,  // For self-recursion
    params: Vec<String>,
    body: Vec<Box<Expr>>,
    captured: Vec<String>, // Captured variables
}
```

**Compilation strategy**:
- Generate closure object on heap
- Capture free variables
- Generate function code
- Return closure pointer

### Phase 2: Control Flow (Week 3)
**Goal**: Add iteration and recursion

#### Step 3: Implement `loop/recur` (2-3 days)
**Why third**: Enables efficient iteration

**What to implement**:
```clojure
(loop [i 0
       sum 0]
  (if (< i 10)
    (recur (inc i) (+ sum i))
    sum))  ;=> 45
```

**Requirements**:
- Tail-call optimization
- Rebind loop variables
- Stack frame reuse

**AST**:
```rust
Expr::Loop {
    bindings: Vec<(String, Box<Expr>)>,
    body: Vec<Box<Expr>>,
}

Expr::Recur {
    args: Vec<Box<Expr>>,
}
```

**Compilation strategy**:
- Mark loop entry point
- `recur` jumps back with new values
- Reuse same stack frame (tail-call)

### Phase 3: Error Handling (Week 4)
**Goal**: Add exception handling

#### Step 4: Implement `try/catch/finally` (2-3 days)
**What to implement**:
```clojure
(try
  (/ 1 0)
  (catch ArithmeticException e
    "Division by zero")
  (finally
    (println "Cleanup")))
```

**Requirements**:
- Exception propagation
- Catch clauses (by type)
- Finally block execution

**AST**:
```rust
Expr::Try {
    body: Vec<Box<Expr>>,
    catches: Vec<CatchClause>,
    finally: Option<Vec<Box<Expr>>>,
}

struct CatchClause {
    exception_type: String,
    binding: String,
    body: Vec<Box<Expr>>,
}
```

**Compilation strategy**:
- Generate exception handlers
- Stack unwinding
- Finally block always runs

#### Step 5: Implement `throw` (1 day)
**What to implement**:
```clojure
(throw (Exception. "error message"))
```

**AST**:
```rust
Expr::Throw {
    exception: Box<Expr>,
}
```

### Phase 4: Advanced (Week 5+)
**Goal**: Complete the feature set

#### Step 6: Implement `var` (1 day)
**What to implement**:
```clojure
(def x 10)
(var x)      ;=> #'user/x
(deref #'x)  ;=> 10
```

**Requirements**:
- Return Var object (not value)
- Support metadata access

## Implementation Strategy for Each Phase

### General Approach
1. **AST First**: Add AST variant
2. **Analyzer**: Add `analyze_*` function
3. **IR**: Add IR instructions if needed
4. **Compiler**: Implement compilation
5. **ARM64 Codegen**: Generate machine code
6. **Tests**: Comprehensive test suite
7. **Documentation**: Update docs

### Testing Strategy
For each special form:
1. **Unit tests**: Test in isolation
2. **Integration tests**: Test with other forms
3. **Edge cases**: Nil, empty, errors
4. **Clojure comparison**: Match official behavior

## Detailed Implementation: `let` (First Priority)

### Step-by-Step Plan

#### 1. Add AST Variant (src/clojure_ast.rs)
```rust
pub enum Expr {
    // ... existing variants

    Let {
        bindings: Vec<(String, Box<Expr>)>,
        body: Vec<Box<Expr>>,
    },
}
```

#### 2. Add Analyzer (src/clojure_ast.rs)
```rust
fn analyze_let(items: &im::Vector<Value>) -> Result<Expr, String> {
    // (let [x 10 y 20] (+ x y))
    //      ^^^^^^^^^  ^^^^^^^^^^
    //      bindings   body

    if items.len() < 3 {
        return Err("let requires bindings and body".to_string());
    }

    // Parse bindings vector
    let bindings_vec = match &items[1] {
        Value::Vector(v) => v,
        _ => return Err("let requires a vector of bindings".to_string()),
    };

    if bindings_vec.len() % 2 != 0 {
        return Err("let bindings must be even (symbol value pairs)".to_string());
    }

    // Parse binding pairs
    let mut bindings = Vec::new();
    for i in (0..bindings_vec.len()).step_by(2) {
        let name = match &bindings_vec[i] {
            Value::Symbol(s) => s.clone(),
            _ => return Err("let binding names must be symbols".to_string()),
        };
        let value = analyze(&bindings_vec[i + 1])?;
        bindings.push((name, Box::new(value)));
    }

    // Parse body expressions
    let mut body = Vec::new();
    for i in 2..items.len() {
        body.push(Box::new(analyze(&items[i])?));
    }

    if body.is_empty() {
        return Err("let requires at least one body expression".to_string());
    }

    Ok(Expr::Let { bindings, body })
}
```

#### 3. Update Pattern Match (src/clojure_ast.rs)
```rust
match name.as_str() {
    "def" => analyze_def(items),
    "let" => analyze_let(items),  // NEW
    "if" => analyze_if(items),
    // ... rest
}
```

#### 4. Add IR Instructions (src/ir.rs)
```rust
pub enum Instruction {
    // ... existing instructions

    // Local variable management
    AllocLocal(String),      // Allocate stack slot for local
    StoreLocal(String, IrValue),  // Store value to local
    LoadLocal(String) -> IrValue, // Load value from local
    FreeLocal(String),       // Free stack slot
}
```

#### 5. Compile Let (src/compiler.rs)
```rust
fn compile_let(
    &mut self,
    bindings: &[(String, Box<Expr>)],
    body: &[Box<Expr>],
) -> Result<IrValue, String> {
    // Extend environment with new scope
    self.push_scope();

    // Compile each binding
    for (name, value_expr) in bindings {
        let value_reg = self.compile(value_expr)?;

        // Store in local variable map
        self.bind_local(name.clone(), value_reg);
    }

    // Compile body (return last expression)
    let mut result = IrValue::Nil;
    for expr in body {
        result = self.compile(expr)?;
    }

    // Clean up scope
    self.pop_scope();

    Ok(result)
}
```

#### 6. Update Var Lookup (src/compiler.rs)
```rust
fn lookup_var(&self, namespace: &Option<String>, name: &str) -> Result<IrValue, String> {
    // Check local bindings first
    if let Some(local_reg) = self.lookup_local(name) {
        return Ok(local_reg);
    }

    // Fall back to global var lookup
    // ... existing code
}
```

#### 7. Tests (tests/test_let.txt)
```clojure
;; Basic let
(let [x 10] x)
;=> 10

;; Multiple bindings
(let [x 10 y 20] (+ x y))
;=> 30

;; Sequential bindings
(let [x 10
      y (+ x 5)
      z (+ y x)]
  z)
;=> 25

;; Shadowing
(def x 100)
(let [x 10] x)
;=> 10
x
;=> 100

;; Nested let
(let [x 10]
  (let [y 20]
    (+ x y)))
;=> 30
```

## Expected Timeline

### Aggressive Schedule (4-5 weeks)
- Week 1: `let` (2 days) + start `fn` (3 days)
- Week 2: Finish `fn` (2 days) + testing (3 days)
- Week 3: `loop/recur` (3 days) + testing (2 days)
- Week 4: `try/catch/finally` + `throw` (4 days) + testing (1 day)
- Week 5: `var` + polish + comprehensive testing

### Realistic Schedule (6-8 weeks)
- Week 1-2: `let` with thorough testing
- Week 3-4: `fn` with closures and multiple arities
- Week 5: `loop/recur`
- Week 6: Exception handling
- Week 7-8: Polish, documentation, edge cases

## Success Criteria

After completing Tier 1 & 2, you should be able to write:

```clojure
;; Factorial with fn and recursion
(def factorial
  (fn [n]
    (loop [i n
           acc 1]
      (if (<= i 1)
        acc
        (recur (dec i) (* acc i))))))

(factorial 5)  ;=> 120

;; Map function
(def map
  (fn [f coll]
    (loop [items coll
           result []]
      (if (empty? items)
        result
        (recur (rest items)
               (conj result (f (first items))))))))

(map (fn [x] (* x x)) [1 2 3 4])
;=> [1 4 9 16]

;; Error handling
(def safe-divide
  (fn [x y]
    (try
      (/ x y)
      (catch ArithmeticException e
        :error))))

(safe-divide 10 2)  ;=> 5
(safe-divide 10 0)  ;=> :error
```

## Decision Points

### Question 1: Destructuring in `let`?
**Decision**: Implement basic `let` first, add destructuring later
**Reason**: Destructuring is complex, get basics working first

### Question 2: Variadic functions in `fn`?
**Decision**: Implement fixed-arity first, add `& args` later
**Reason**: Fixed-arity covers 90% of use cases

### Question 3: Exception types?
**Decision**: Single catch-all exception type first
**Reason**: Avoid complex type system initially

## Next Immediate Steps

1. ✅ Read this roadmap
2. 🔲 Decide on timeline (aggressive vs realistic)
3. 🔲 Start with `let` implementation
4. 🔲 Write comprehensive tests
5. 🔲 Move to `fn`
6. 🔲 Continue down the priority list

## Resources

- [Clojure Special Forms Reference](https://clojure.org/reference/special_forms)
- [ClojureScript Compiler](https://github.com/clojure/clojurescript) - Similar goals
- Our existing implementations: `def`, `if`, `do`, `binding`

---

**Ready to start? Let's implement `let` first!** 🚀
