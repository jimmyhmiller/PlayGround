# Modular Import System ✨

## Split Your Compiler Across Multiple Files!

The import system lets you organize your compiler into reusable modules.

## Quick Example

### File Structure

```
bootstrap-modular.lisp          # Main bootstrap
dialects/
  ├─ lisp-core.lisp            # Dialect definition
  ├─ optimizations.lisp        # Optimization patterns
  └─ lowering.lisp             # Lowering transforms
```

### bootstrap-modular.lisp

```lisp
;; Just import what you need!
(import lisp-core)
(import optimizations)
(import lowering)

;; Done! Everything is loaded
```

### dialects/lisp-core.lisp

```lisp
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
```

### dialects/optimizations.lisp

```lisp
(defpdl-pattern constant-fold-add
  :benefit 10
  :match (...)
  :rewrite (...))

(deftransform optimize
  (transform.sequence
    (transform.apply-patterns
      (use-pattern constant-fold-add))))
```

### dialects/lowering.lisp

```lisp
(deftransform lower-to-arith
  (transform.sequence
    (let [adds (transform.match :ops ["lisp.add"])]
      (transform.apply-patterns :to adds
        (use-pattern add-lowering)))))

(defpdl-pattern add-lowering
  :match (pdl.operation "lisp.add" ...)
  :rewrite (pdl.operation "arith.addi" ...))
```

## Run It

```bash
cargo run --example modular_demo
```

## Output

```
✅ Bootstrap loaded successfully!

📦 Dialect: lisp
   Operations (6):
     • constant - Immutable constant value
     • add - Pure functional addition
     • sub - Pure functional subtraction
     • mul - Pure functional multiplication
     • if - Conditional expression
     • call - Tail-call optimizable function call

🔄 Transform: optimize
🔄 Transform: lower-to-arith

🎨 Patterns: 7
   • constant-fold-add (benefit: 10)
   • constant-fold-mul (benefit: 10)
   • eliminate-dead-code (benefit: 5)
   • add-lowering (benefit: 1)
   • sub-lowering (benefit: 1)
   • mul-lowering (benefit: 1)
   • constant-lowering (benefit: 1)
```

## How It Works

### Import Resolution

```
(import lisp-core)
    ↓
Search paths:
  • ./lisp-core
  • ./lisp-core.lisp
  • ./dialects/lisp-core
  • ./dialects/lisp-core.lisp
  • ./lib/lisp-core.lisp
    ↓
Found: dialects/lisp-core.lisp
    ↓
Load and evaluate once
    ↓
Register dialect
```

### Features

✅ **Search Paths**
- Current directory: `./`
- Dialects directory: `./dialects/`
- Library directory: `./lib/`

✅ **File Extensions**
- No extension: `lisp-core`
- `.lisp` extension: `lisp-core.lisp`
- `.mlir-lisp` extension: `lisp-core.mlir-lisp`

✅ **Load Once**
- Files are only loaded once
- Prevents circular dependencies
- Returns `already-loaded` if reimported

✅ **Custom Search Paths**
```rust
compiler.add_search_path("/path/to/my/dialects".to_string());
```

## Benefits

### 📁 Separation of Concerns

**Before (monolithic):**
```lisp
;; bootstrap.lisp - 200 lines
(defirdl-dialect lisp ...)
(defpdl-pattern opt1 ...)
(defpdl-pattern opt2 ...)
(deftransform lower ...)
;; ... everything in one file
```

**After (modular):**
```lisp
;; bootstrap-modular.lisp - 3 lines
(import lisp-core)      ; 50 lines
(import optimizations)  ; 70 lines
(import lowering)       ; 80 lines
```

### 🔄 Reusability

Share modules across projects:

```lisp
;; project1/bootstrap.lisp
(import lisp-core)
(import optimizations)

;; project2/bootstrap.lisp
(import lisp-core)
(import my-custom-opts)
```

### 🧪 Testability

Test modules independently:

```lisp
;; test-optimizations.lisp
(import lisp-core)
(import optimizations)
;; Test just the optimizations
```

### 👥 Collaboration

Team members work on separate files:

```
Alice works on:   dialects/lisp-core.lisp
Bob works on:     dialects/optimizations.lisp
Carol works on:   dialects/lowering.lisp
```

## Custom Bootstraps

### Minimal Bootstrap

```lisp
;; minimal.lisp
(import lisp-core)
;; Just the dialect, no transforms
```

### Optimization Only

```lisp
;; opt-only.lisp
(import lisp-core)
(import optimizations)
;; No lowering
```

### Full Stack

```lisp
;; full.lisp
(import lisp-core)
(import optimizations)
(import lowering)
(import my-custom-passes)
```

## Creating Your Own Module

### 1. Create a file in `dialects/`

```lisp
;; dialects/my-lang.lisp
(defirdl-dialect my-lang
  :namespace "my"

  (defirdl-op hello
    :summary "Hello operation"
    :results [(result AnyType)]))
```

### 2. Import it

```lisp
;; my-bootstrap.lisp
(import my-lang)
```

### 3. Use it

```rust
compiler.load_file("my-bootstrap.lisp")?;
compiler.eval_string("(list-dialects)")?;
// => ["my-lang"]
```

## Advanced: Search Paths

```rust
let mut compiler = SelfContainedCompiler::new(&context);

// Add custom search paths
compiler.add_search_path("/usr/local/lib/mlir-lisp".to_string());
compiler.add_search_path("~/.mlir-lisp/dialects".to_string());

// Now imports search these paths too
compiler.load_file("bootstrap.lisp")?;
```

## Comparison

### Traditional MLIR

```cpp
// Must compile everything together
#include "Dialect1.h"
#include "Dialect2.h"
#include "Passes.h"
// Recompile to add new dialects
```

### Our System

```lisp
;; Just import what you need
(import dialect1)
(import dialect2)
;; No recompilation!
```

## Implementation

**Location**: `src/self_contained.rs:155-170`

```rust
fn load_import(&mut self, name: &str) -> Result<Value, String> {
    let extensions = vec!["", ".lisp", ".mlir-lisp"];

    for search_path in &self.search_paths.clone() {
        for ext in &extensions {
            let file_path = format!("{}/{}{}", search_path, name, ext);

            if std::path::Path::new(&file_path).exists() {
                return self.load_file(&file_path);
            }
        }
    }

    Err(format!("Could not find import '{}'", name))
}
```

## Try It!

```bash
# Run the modular demo
cargo run --example modular_demo

# Check the file structure
ls dialects/

# Create your own module
echo "(defirdl-dialect test ...)" > dialects/test.lisp

# Import it
echo "(import test)" > my-bootstrap.lisp
```

**The import system makes your compiler truly modular!** 🎉
