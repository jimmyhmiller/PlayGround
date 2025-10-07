# Macro System Examples

## ✅ Currently Working

### Basic Identity Macro
```clojure
(defmacro identity [x] x)

(def result (: Int) (identity 42))
result  ; => 42
```

### Quote/Unquote Syntax

#### Single Unquote
```clojure
(defmacro add1 [x] `(+ ~x 1))

(def result (: Int) (add1 5))  ; expands to (+ 5 1) => 6
```

#### Multiple Unquotes
```clojure
(defmacro add-three [a b c] `(+ ~a ~b ~c))

(def result (: Int) (add-three 10 20 30))  ; => 60
```

#### Mixed Quoted and Unquoted
```clojure
(defmacro add-with-constants [x] `(+ 10 ~x 20))

(def result (: Int) (add-with-constants 7))  ; expands to (+ 10 7 20) => 37
```

#### Nested Structures
```clojure
(defmacro double-then-add [x y] `(+ (* 2 ~x) ~y))

(def result (: Int) (double-then-add 5 3))  ; expands to (+ (* 2 5) 3) => 13
```

#### Unquote-Splicing
```clojure
(defmacro sum-list [xs] `(+ ~@xs))

(def result (: Int) (sum-list (1 2 3 4 5)))  ; expands to (+ 1 2 3 4 5) => 15
```

### Conditional Macro (unless)
```clojure
(defmacro unless [condition body]
  (if condition 0 body))

(def result1 (: Int) (unless false 42))  ; => 42
(def result2 (: Int) (unless true 99))   ; => 0
(+ result1 result2)  ; => 42
```

### Arithmetic Macros
```clojure
(defmacro double [x] (+ x x))
(defmacro square [x] (* x x))

(def a (: Int) (double 10))   ; => 20
(def b (: Int) (square 7))    ; => 49
```

### Nested Macro Calls
```clojure
(defmacro add1 [x] (+ x 1))

(def result (: Int) (add1 (add1 (add1 10))))  ; => 13
```

### Macro Using Another Macro
```clojure
(defmacro double [x] (+ x x))
(defmacro quadruple [x] (double (double x)))

(def result (: Int) (quadruple 3))  ; => 12
```

## How It Works

1. **Macro Definition**: `(defmacro name [params] body)`
   - Defines a compile-time transformation
   - Stores in MacroEnv during expansion phase

2. **Macro Expansion**: Happens before type checking
   - Pipeline: Parse → **Expand Macros** → Type Check → Codegen
   - Template substitution: replaces parameters with actual arguments
   - Recursive expansion: macros can call other macros

3. **Syntax-Quote Templates**: Reader syntax for templates
   - `` `expr `` or `(syntax-quote expr)` - Quote structure
   - `~x` or `(unquote x)` - Substitute parameter value
   - `~@xs` or `(unquote-splicing xs)` - Splice list into surrounding context

   **Note**: Reader syntax (backtick/tilde) is preferred for conciseness

4. **Example Expansion**:
   ```clojure
   (defmacro add1 [x] `(+ ~x 1))
   (add1 5)

   ; Step 1: Substitute x with 5 in template `(+ ~x 1)
   ; Step 2: Unquote ~x becomes 5
   ; Expands to: (+ 5 1)

   ; Which becomes C code:
   (5 + 1)  // = 6
   ```

5. **Unquote-Splicing Example**:
   ```clojure
   (defmacro sum-list [xs] `(+ ~@xs))
   (sum-list (1 2 3))

   ; Step 1: Substitute xs with (1 2 3)
   ; Step 2: Unquote-splice ~@xs splices elements into list
   ; Expands to: (+ 1 2 3)
   ```

## Current Features

- ✅ Simple template substitution works
- ✅ Quote/unquote syntax (`` ` ``, `~`, `~@`)
- ✅ Nested structures and vectors
- ✅ Error handling for misplaced unquote
- ✅ Macros using other macros
- ❌ No `macroexpand`/`macroexpand-all` for debugging
- ❌ No gensym for hygiene (manual unique naming required)
- ❌ No variadic macros (`& rest`)
- ❌ REPL doesn't support macros yet

## Planned Features

### 1. Macro Introspection (Debug Tools)
```clojure
;; Expand macro once
(macroexpand '(add1 5))
; => (+ 5 1)

;; Expand all macros recursively
(defmacro quadruple [x] (double (double x)))
(macroexpand-all '(quadruple 3))
; => (+ (+ 3 3) (+ 3 3))
```

### 2. REPL Integration
- Define and use macros interactively
- Redefinition support
- Persistent macro environment

### 3. Gensym (Optional)
- Generate unique symbols for hygienic macros
- Avoid variable capture

### 4. Variadic Macros (Future)
```clojure
(defmacro when [condition & body]
  `(if ~condition (do ~@body) nil))
```

## Testing

All macro functionality is tested in:
- `src/macro_expander.zig` - Unit tests for expansion logic
- `src/macro_comprehensive_tests.zig` - Integration tests
- `scratch/test_macro_*.lisp` - Example programs

Run tests with: `zig test src/test_all.zig`
