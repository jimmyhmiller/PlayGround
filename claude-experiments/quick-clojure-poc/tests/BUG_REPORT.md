# Bug Report: Bugs Found in Quick Clojure PoC

## Summary

Through systematic testing, the following bugs and missing features were discovered
in the existing implementation. This report distinguishes between **BUGS** (things
that are documented as implemented but don't work correctly) and **NOT IMPLEMENTED**
(features that are simply missing).

---

## BUGS (Broken Functionality)

### BUG 1: `count` on lists created with `(list ...)` fails
**Severity: HIGH**

```clojure
(def lst (list 1 2 3))
(count lst)  ;; Throws: "IllegalArgumentException: No implementation of method..."
```

The ICounted protocol is not implemented for the Cons/list type. `first` and `rest`
work correctly on lists, but `count` fails with a protocol lookup error.

---

### BUG 2: `conj` on sets returns a vector instead of a set
**Severity: HIGH**

```clojure
(conj #{1 2} 3)  ;; Returns [1 2 3] (vector) instead of #{1 2 3} (set)
```

The `conj` operation on sets incorrectly returns a vector containing the elements
instead of a proper set. This breaks set semantics completely.

---

### BUG 3: `assoc` and `dissoc` print object addresses instead of map contents
**Severity: MEDIUM**

```clojure
(println (assoc {:a 1} :b 2))
;; Prints: #<clojure.core/PersistentHashMap@3000bf570>
;; Should print: {:a 1 :b 2}
```

The operations work correctly (get/contains? work after assoc), but the printing
function doesn't properly format the result. The map is displayed as an opaque
object reference.

---

### BUG 4: `binding` with dynamic vars causes codegen error
**Severity: HIGH**

```clojure
(def ^:dynamic *d* 100)
(binding [*d* 200] (println *d*))
;; Error: Codegen error: Codegen received TaggedConstant(...) where a register was expected.
```

Dynamic variable binding compiles but fails at code generation with an internal error.
The compiler doesn't properly handle TaggedConstants in the CallWithSaves instruction.

---

## NOT IMPLEMENTED (Missing Functions)

The following core Clojure functions are documented/implied but not implemented:

### Core Functions
- `map` - Higher-order map function
- `filter` - Higher-order filter function
- `apply` - Apply function to argument list
- `disj` - Remove element from set
- `even?` - Check if number is even
- `odd?` - Check if number is odd
- `str` - String concatenation
- `name` - Get name of keyword/symbol as string
- `keyword` - Create keyword from string
- `abs` - Absolute value
- `min` - Minimum of two numbers
- `max` - Maximum of two numbers
- `mod` - Modulo operation
- `rem` - Remainder operation

---

## WORKING FEATURES (Verified by Tests)

The following features work correctly:

### Arithmetic (Binary Only)
- `+`, `-`, `*`, `/` (binary operators only)
- Large number arithmetic (up to ~2 billion)
- Negative numbers
- Integer division

### Comparisons
- `<`, `>`, `=`, `<=`, `>=`, `not=`
- Comparing negative numbers
- Nil and boolean comparisons

### Vectors
- `first`, `rest`, `count`, `nth`
- `conj` (to vectors)
- Empty vector operations
- Nested vector access

### Lists
- `first`, `rest` (NOT count!)
- Creation with `(list ...)`

### Maps
- `get` with default values
- `count`, `keys`, `vals`
- `contains?`
- Keywords as lookup functions (`:key map`)
- Nested map access
- Map with nil values (correctly distinguishes missing key vs nil value)

### Sets
- `count`, `contains?` work
- (but NOT `conj` or `disj`)

### Functions
- Single-arity, multi-arity, variadic
- Closures with proper capture
- Recursive functions
- Zero-arity functions
- Functions returning nil/false

### Control Flow
- `if` (with proper truthiness: nil and false are falsy, 0/[]/{}/"" are truthy)
- `let` with shadowing and sequential bindings
- `loop`/`recur` for tail recursion
- `do` blocks
- `when`, `cond` macros
- `->` threading macro

### Try/Catch
- Exception handling works
- Pre-conditions with `:pre`
- Custom `throw` with catch

### Boolean Operations
- `not`, `and`, `or`
- Short-circuit evaluation

### Predicates
- `nil?`, `true?`, `false?`
- `vector?`, `list?`, `map?`
- `zero?`, `number?`, `string?`, `keyword?`

### Macros
- `defmacro` works
- Syntax quote, unquote, unquote-splicing
- `gensym` for hygiene

### Deftype/Protocol
- `deftype` with field access `.-field`
- Mutable fields with `^:mut` and `set!`
- `defprotocol` and `extend-type`
- Protocol dispatch works

### Misc
- `cons` (to nil, vectors, lists)
- `into` (to vectors)
- `reduce` with initial value
- `list*`
- `symbol` function
- `inc`, `dec`
- `bit-and`, `bit-or`, `bit-xor`
- Keyword equality

---

## Test Files Created

1. `tests/test_bugs_working.clj` - Tests all working functionality
2. `tests/test_find_bugs.clj` - Initial exploratory tests
3. `tests/test_bugs_isolated.clj` - Isolated test cases
4. `tests/test_comprehensive.clj` - Comprehensive test suite

---

## Recommendations

1. **High Priority**: Fix `count` for lists - this breaks many common patterns
2. **High Priority**: Fix `conj` for sets - completely broken set semantics
3. **High Priority**: Fix `binding` codegen - dynamic vars don't work
4. **Medium Priority**: Fix `assoc`/`dissoc` printing
5. **Medium Priority**: Implement missing core functions (map, filter, str, etc.)
