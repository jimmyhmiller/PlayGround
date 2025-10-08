# Self-Contained Meta-Circular Compiler - Quick Example

## Everything in One Lisp File! 🚀

### bootstrap.lisp - The Entire Compiler

```lisp
;; ============================================================================
;; This single file defines the ENTIRE compiler!
;; ============================================================================

;; 1. DEFINE THE DIALECT
(defirdl-dialect lisp
  :namespace "lisp"

  (defirdl-op constant
    :attributes [(value IntegerAttr)]
    :results [(result AnyInteger)]
    :traits [Pure NoMemoryEffect])

  (defirdl-op add
    :operands [(lhs AnyInteger) (rhs AnyInteger)]
    :results [(result AnyInteger)]
    :traits [Pure Commutative NoMemoryEffect]))

;; 2. DEFINE OPTIMIZATIONS
(defpdl-pattern constant-fold-add
  :benefit 10
  :match
  (let [c1 (pdl.operation "lisp.constant")
        c2 (pdl.operation "lisp.constant")
        add (pdl.operation "lisp.add" :operands [c1 c2])]
    add)
  :rewrite
  (pdl.operation "lisp.constant" :value (+ c1.value c2.value)))

;; 3. DEFINE LOWERING
(deftransform lower-to-arith
  (transform.sequence
    (let [adds (transform.match :ops ["lisp.add"])]
      (transform.apply-patterns :to adds
        (use-pattern add-lowering)))))

(defpdl-pattern add-lowering
  :match (pdl.operation "lisp.add" :operands [lhs rhs])
  :rewrite (pdl.operation "arith.addi" :operands [lhs rhs]))

;; Done! The compiler is defined!
```

## Run It

```bash
cargo run --example self_contained_demo
```

## Output

```
✅ Bootstrap complete!

Registered Dialects:
  • lisp

Registered Transforms:
  • lower-to-arith

Dialect: lisp
  Operations: 6
    • constant: Immutable constant value
    • add: Pure functional addition
    • sub: Pure functional subtraction
    • mul: Pure functional multiplication
    • if: Conditional expression
    • call: Tail-call optimizable function call
```

## What Just Happened?

1. ✅ **Loaded `bootstrap.lisp`** - Single Lisp file
2. ✅ **Defined a dialect** - `lisp` with operations
3. ✅ **Defined optimizations** - Constant folding patterns
4. ✅ **Defined lowering** - Transform to `arith` dialect
5. ✅ **Everything available** - Queryable from Lisp

## Introspection from Lisp

```lisp
(list-dialects)           ;; => ["lisp"]
(list-transforms)         ;; => ["lower-to-arith"]
(list-patterns)           ;; => ["constant-fold-add", "add-lowering"]
(get-dialect "lisp")      ;; => Full dialect info
```

## The Magic

```rust
// Just 3 lines of Rust!
let context = Context::new();
let mut compiler = SelfContainedCompiler::new(&context);
compiler.load_file("bootstrap.lisp")?;

// Everything else is Lisp!
```

## Why This Is Revolutionary

### Traditional MLIR

❌ Define dialect in **C++ TableGen**
❌ Write passes in **C++**
❌ Recompile to add features
❌ Limited introspection

### Our System

✅ Define dialect in **Lisp**
✅ Write transforms in **Lisp**
✅ Load at runtime - no recompilation
✅ Full introspection from Lisp

## The Meta-Circular Loop

```
bootstrap.lisp defines:
    ↓
  Dialect "lisp"
    ↓
  Which compiles:
    ↓
  Programs in Lisp
    ↓
  Including bootstrap.lisp itself!
```

**The compiler compiles itself!** 🎉

## Next: Write Your Program

```lisp
;; my_program.lisp
(defn compute [] i32
  (+ (* 10 20) 30))

;; The compiler will:
;; 1. Parse to lisp.* operations
;; 2. Apply constant-fold-mul: (* 10 20) → 200
;; 3. Apply constant-fold-add: (+ 200 30) → 230
;; 4. Lower to arith.constant 230
```

## Try It Now!

```bash
# 1. See the bootstrap file
cat bootstrap.lisp

# 2. Run the demo
cargo run --example self_contained_demo

# 3. Modify bootstrap.lisp and run again!
```

**Everything is Lisp. Nothing is hidden. The compiler is defined in the language it compiles.**

That's the meta-circular ideal! ✨
