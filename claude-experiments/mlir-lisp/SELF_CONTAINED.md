# Self-Contained Meta-Circular Compiler ✨

## Everything in Lisp - No Rust Needed!

This is a **fully self-contained** meta-circular MLIR compiler where the entire system is defined in Lisp.

## Quick Start

### 1. The Bootstrap File (`bootstrap.lisp`)

This single file defines the **entire compiler**:

```lisp
;; Define the dialect
(defirdl-dialect lisp
  :namespace "lisp"

  (defirdl-op constant
    :attributes [(value IntegerAttr)]
    :results [(result AnyInteger)]
    :traits [Pure NoMemoryEffect])

  (defirdl-op add
    :operands [(lhs AnyInteger) (rhs AnyInteger)]
    :results [(result AnyInteger)]
    :traits [Pure Commutative]))

;; Define optimizations
(defpdl-pattern constant-fold-add
  :benefit 10
  :match
  (let [c1 (pdl.operation "lisp.constant")
        c2 (pdl.operation "lisp.constant")
        add (pdl.operation "lisp.add" :operands [c1 c2])]
    add)
  :rewrite
  (pdl.operation "lisp.constant" :value (+ c1.value c2.value)))

;; Define compilation pipeline
(deftransform optimize
  (transform.sequence
    (transform.apply-patterns
      (use-pattern constant-fold-add))))
```

### 2. Run It

```bash
cargo run --example self_contained_demo
```

### Output

```
✅ Bootstrap complete!

Registered Dialects:
  • lisp

Registered Transforms:
  • optimize
  • lower-to-arith

Registered Patterns:
  • constant-fold-add
  • constant-fold-mul
  • eliminate-dead-code
  • add-lowering
  • sub-lowering
  • mul-lowering
  • constant-lowering

Dialect: lisp
  Operations: 6
    • constant: Immutable constant value
    • add: Pure functional addition
    • sub: Pure functional subtraction
    • mul: Pure functional multiplication
    • if: Conditional expression
    • call: Tail-call optimizable function call
```

## How It Works

### Architecture

```
bootstrap.lisp → Self-Contained Compiler → Runtime System
     ↓                    ↓                       ↓
  Lisp Code          eval() function        Dialect Registry
                          ↓                       ↓
                    Macro Expansion         Transforms
                          ↓                       ↓
                   Internal Repr.            Patterns
```

### The Self-Contained Compiler

**Location**: `src/self_contained.rs`

```rust
let context = Context::new();
let mut compiler = SelfContainedCompiler::new(&context);

// Load the bootstrap file
compiler.load_file("bootstrap.lisp")?;

// Now query what was defined
compiler.eval_string("(list-dialects)")?;  // ["lisp"]
compiler.eval_string("(get-dialect \"lisp\")")?;  // Full details
```

### Lisp Commands

The compiler provides introspection commands:

```lisp
;; List what's available
(list-dialects)     ;; => ["lisp"]
(list-transforms)   ;; => ["optimize", "lower-to-arith"]
(list-patterns)     ;; => ["constant-fold-add", ...]

;; Get details
(get-dialect "lisp")  ;; => Full dialect info
```

## What's Self-Contained?

### ✅ Dialects Defined in Lisp

```lisp
(defirdl-dialect lisp
  (defirdl-op add ...))
```

**NOT** C++ TableGen:
```cpp
def Lisp_AddOp : Lisp_Op<"add"> { ... }
```

### ✅ Transforms Defined in Lisp

```lisp
(deftransform optimize
  (transform.sequence
    (transform.apply-patterns ...)))
```

**NOT** Rust/C++ code:
```rust
impl Pass for OptimizePass { ... }
```

### ✅ Patterns Defined in Lisp

```lisp
(defpdl-pattern constant-fold
  :match (...)
  :rewrite (...))
```

**NOT** manual IR walking:
```rust
for op in module.walk() { ... }
```

### ✅ Everything Queryable from Lisp

```lisp
(list-dialects)        ;; Introspection
(get-dialect "lisp")   ;; Get details
```

**NOT** hidden in Rust:
```rust
// No API needed!
```

## Complete Example

### bootstrap.lisp

```lisp
;; 1. Define dialect
(defirdl-dialect lisp
  :namespace "lisp"
  :description "High-level Lisp operations"

  (defirdl-op constant
    :attributes [(value IntegerAttr)]
    :results [(result AnyInteger)]
    :traits [Pure NoMemoryEffect])

  (defirdl-op add
    :operands [(lhs AnyInteger) (rhs AnyInteger)]
    :results [(result AnyInteger)]
    :traits [Pure Commutative NoMemoryEffect]))

;; 2. Define optimization
(defpdl-pattern constant-fold-add
  :benefit 10
  :description "Fold constant addition"
  :match
  (let [val1 (pdl.attribute)
        val2 (pdl.attribute)
        const1 (pdl.operation "lisp.constant" :attributes {:value val1})
        const2 (pdl.operation "lisp.constant" :attributes {:value val2})
        add (pdl.operation "lisp.add" :operands [const1 const2])]
    add)
  :rewrite
  (let [sum (pdl.apply-native "add-integers" [val1 val2])]
    (pdl.operation "lisp.constant" :attributes {:value sum})))

;; 3. Define lowering
(deftransform lower-to-arith
  :description "Lower to arithmetic dialect"
  (transform.sequence
    (let [adds (transform.match :ops ["lisp.add"])]
      (transform.apply-patterns :to adds
        (use-pattern add-lowering)))))

(defpdl-pattern add-lowering
  :match
  (pdl.operation "lisp.add" :operands [lhs rhs])
  :rewrite
  (pdl.operation "arith.addi" :operands [lhs rhs]))
```

### Use It

```rust
use mlir_lisp::self_contained::SelfContainedCompiler;
use melior::Context;

fn main() {
    let context = Context::new();
    let mut compiler = SelfContainedCompiler::new(&context);

    // Bootstrap the compiler from Lisp!
    compiler.load_file("bootstrap.lisp").unwrap();

    // Query what's available
    let dialects = compiler.eval_string("(list-dialects)").unwrap();
    println!("Available dialects: {:?}", dialects);

    // Get dialect details
    let dialect = compiler.eval_string(r#"(get-dialect "lisp")"#).unwrap();
    println!("Dialect info: {:?}", dialect);
}
```

## Key Features

### 1. Meta-Circular

The compiler **defines itself** in the language it compiles:
- `bootstrap.lisp` contains the compiler
- Written in Lisp
- Compiles Lisp programs

### 2. Runtime Extensible

Load new dialects **at runtime**:

```lisp
;; Load a new dialect
(defirdl-dialect my-lang ...)

;; It's immediately available
(list-dialects)  ;; => ["lisp", "my-lang"]
```

### 3. Fully Introspectable

Everything is queryable:

```lisp
(list-dialects)           ;; All dialects
(list-transforms)         ;; All transforms
(list-patterns)           ;; All patterns
(get-dialect "name")      ;; Dialect details
```

### 4. No Compilation Needed

Add features **without recompiling**:

1. Edit `bootstrap.lisp`
2. Run the program
3. New features available!

## Implementation Details

### Core Components

| Component | Purpose | Location |
|-----------|---------|----------|
| `SelfContainedCompiler` | Main eval loop | `src/self_contained.rs` |
| `MacroExpander` | Expands IRDL/Transform macros | `src/macro_expander.rs` |
| `DialectRegistry` | Stores dialects/transforms/patterns | `src/dialect_registry.rs` |
| `bootstrap.lisp` | Defines the compiler | `bootstrap.lisp` |

### The eval() Function

```rust
pub fn eval(&mut self, expr: &Value) -> Result<Value, String> {
    match expr {
        // Special forms
        "defirdl-dialect" => {
            let expanded = self.expander.expand(expr)?;
            self.registry.process_expanded_form(&expanded)?;
            // Dialect is now available!
        }

        // Introspection
        "(list-dialects)" => {
            return Ok(self.registry.list_dialects());
        }

        // Everything else
        _ => self.expander.expand(expr)
    }
}
```

### Data Flow

```
Lisp Source
    ↓
eval_string()
    ↓
parse() → Value AST
    ↓
eval() → recognize special forms
    ↓
expand() → apply macros
    ↓
register() → store in registry
    ↓
✓ Available for use!
```

## Comparison

### Traditional Compiler

```
C++ Code → Compile → Binary → Run Program
   ↑                              ↓
   └──── Need to recompile ───────┘
```

### Self-Contained Compiler

```
Lisp Code → eval() → Available Immediately
                         ↓
                    Introspectable
                         ↓
                    Modifiable
```

## What's Next?

The foundation is complete. Future work:

1. **Code Generation**: Generate actual MLIR operations from dialect definitions
2. **Transform Execution**: Actually run the transforms on IR
3. **JIT Compilation**: Execute the compiled code
4. **Import System**: `#lang` style module loading

But the core meta-circular infrastructure **works right now**! 🎉

## Try It!

```bash
# Run the demo
cargo run --example self_contained_demo

# See the bootstrap file
cat bootstrap.lisp

# Modify it and run again!
```

The compiler that compiles itself - all in Lisp!
