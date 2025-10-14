# General Dialect Capability

## The Key Insight

**We have a GENERAL capability to work with ANY dialect, not special-case code for specific dialects.**

## How It Works

### 1. Define ANY dialect in Lisp

```lisp
(defirdl-dialect my-dialect
  :namespace "myns"
  :description "Any dialect you want"

  (defirdl-op my-op
    :summary "Any operation"
    :operands [(x I32)]
    :results [(result I32)]))
```

### 2. Write programs using that dialect

```lisp
(defn foo [] i32
  (myns.my-op (myns.constant 42)))
```

### 3. The system emits the operations

Our ExprCompiler sees `myns.my-op` and:
- Looks it up in the dialect registry
- Emits the MLIR operation
- Done!

**No special Rust code needed for this dialect.**

## This Applies to Transform Dialect Too!

Transform dialect is just another dialect. We can write transform operations in Lisp:

```lisp
;; Define PDL patterns
(pdl.pattern
  :name "my_pattern"
  :benefit 1
  :body
  (let [val (pdl.attribute)
        op (pdl.operation "myns.constant" :attrs {:value val})]
    (pdl.rewrite op
      (pdl.operation "arith.constant" :attrs {:value val}))))

;; Define transform sequences
(transform.sequence
  :attrs {:failure_propagation_mode "propagate"}
  :body
  (fn [module]
    (let [ops (transform.structured.match :ops ["myns.constant"] :in module)]
      (transform.apply_patterns :to ops :patterns [my_pattern]))))
```

These are just MLIR operations! Our system emits them like any other operation.

## The Only Transform-Specific Code

There's only ONE piece of transform-specific Rust code:

```rust
// src/transform_interpreter.rs
pub fn apply_transform(
    context: &Context,
    transform_module: &Module,  // Contains transform.* ops we wrote in Lisp
    target_module: &Module,     // The module to transform
) -> Result<(), String> {
    // Call MLIR's transform interpreter
    // (Once melior exposes it)
}
```

That's it! This function:
1. Takes the transform IR we generated from Lisp
2. Invokes MLIR's interpreter to execute it
3. Returns the transformed module

**No pattern-specific code. No dialect-specific code. Just one interpreter invocation.**

## Architecture Summary

```
Lisp Source
    ↓
Parser (general)
    ↓
Macro Expander (general)
    ↓
ExprCompiler (general - works with ANY dialect)
    ↓
MLIR IR (includes transform.* operations if we wrote them)
    ↓
Transform Interpreter (one function call)
    ↓
Transformed IR
    ↓
Lower to LLVM (standard MLIR passes)
    ↓
JIT Execute
```

Every step is general! No special cases!

## What Makes This Powerful

1. **Extensible**: Define any dialect, use it immediately
2. **No Recompilation**: New dialects don't require rebuilding the compiler
3. **Meta-Circular**: Transforms are just IR, written in the same language
4. **Debuggable**: Inspect generated transform IR
5. **Composable**: Combine transforms, dialects freely

## Current Status

✅ Can define dialects in Lisp
✅ Can emit operations from those dialects
✅ Can write transform.* and pdl.* operations in Lisp
🔄 Need to call MLIR transform interpreter (waiting on melior support)

The foundation is complete and fully general!
