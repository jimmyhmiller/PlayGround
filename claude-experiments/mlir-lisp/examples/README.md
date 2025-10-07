# MLIR-Lisp Examples

## Running Examples

Run all examples with the test script:
```bash
./run_examples.sh
```

Or run individual examples:
```bash
# Simple constant return
cargo run --release examples/simple.lisp

# Integer arithmetic
cargo run --release examples/add.lisp
cargo run --release examples/subtract.lisp
cargo run --release examples/multiply.lisp
cargo run --release examples/divide.lisp
cargo run --release examples/complex_arithmetic.lisp

# Floating point
cargo run --release examples/float_add.lisp
cargo run --release examples/float_multiply.lisp

# Control flow
cargo run --release examples/simple_branch.lisp
cargo run --release examples/conditional_simple.lisp
```

## simple.lisp

Returns the constant 42.

```lisp
(op arith.constant
    :attrs {:value 42}
    :results [i32]
    :as %result)

(op func.return
    :operands [%result])
```

**Output:** `42`

## add.lisp

Adds two numbers (10 + 32).

```lisp
(op arith.constant
    :attrs {:value 10}
    :results [i32]
    :as %ten)

(op arith.constant
    :attrs {:value 32}
    :results [i32]
    :as %thirty_two)

(op arith.addi
    :operands [%ten %thirty_two]
    :results [i32]
    :as %result)

(op func.return
    :operands [%result])
```

**Output:** `42` (Note: LLVM optimizes this to a constant at compile time!)

## multiply.lisp

Multiplies two numbers (10 * 15).

**Output:** `150`

## subtract.lisp

Subtracts two numbers (50 - 42).

**Output:** `8`

## divide.lisp

Divides two numbers (84 / 12) using signed integer division.

**Output:** `7`

## complex_arithmetic.lisp

Demonstrates chained operations: `(10 + 5) * 2 - 3 = 27`

Shows how SSA values flow through multiple operations. LLVM's optimizer can constant-fold this entire expression at compile time!

**Output:** `27`

## float_add.lisp & float_multiply.lisp

Floating point examples demonstrating f64 arithmetic operations.

**float_add.lisp Output:** `5.7` (3.14 + 2.56)
**float_multiply.lisp Output:** `15.0` (3.0 * 5.0)

## simple_branch.lisp

Demonstrates unconditional branch with block arguments:

```lisp
(block entry []
  (op arith.constant :attrs {:value 42} :results [i32] :as %result)
  (op cf.br :dest exit_block :args [%result]))

(block exit_block [i32]
  (op func.return :operands [^0]))
```

**Output:** `42`

## conditional_simple.lisp

Demonstrates conditional branching (if-then-else) using `cf.cond_br` with a pre-computed boolean:

```lisp
(block entry []
  (op arith.constant :attrs {:value 1} :results [i1] :as %condition)
  (op cf.cond_br :operands [%condition] :true then_block :false else_block))

(block then_block []
  (op arith.constant :attrs {:value 100} :results [i32] :as %result_true)
  (op cf.br :dest exit_block :args [%result_true]))

(block else_block []
  (op arith.constant :attrs {:value 200} :results [i32] :as %result_false)
  (op cf.br :dest exit_block :args [%result_false]))

(block exit_block [i32]
  (op func.return :operands [^0]))
```

**Output:** `100` (because condition is true)

## if_then_else.lisp

Complete if-then-else with comparison using `arith.cmpi`:

```lisp
(block entry []
  (op arith.constant :attrs {:value 10} :results [i32] :as %ten)
  (op arith.constant :attrs {:value 5} :results [i32] :as %five)

  ;; Compare: is 10 > 5?
  (op arith.cmpi :attrs {:predicate "sgt"} :operands [%ten %five] :results [i1] :as %cond)

  (op cf.cond_br :operands [%cond] :true then_block :false else_block))

(block then_block []
  (op arith.constant :attrs {:value 42} :results [i32] :as %result_true)
  (op cf.br :dest exit_block :args [%result_true]))

(block else_block []
  (op arith.constant :attrs {:value 0} :results [i32] :as %result_false)
  (op cf.br :dest exit_block :args [%result_false]))

(block exit_block [i32]
  (op func.return :operands [^0]))
```

**Output:** `42` (because 10 > 5 is true)

## function_simple.lisp

Simple function that adds two numbers and calls it from main:

```lisp
(defn add [x:i32 y:i32] i32
  (op arith.addi :operands [x y] :results [i32] :as %result)
  (op func.return :operands [%result]))

(defn main [] i32
  (op arith.constant :attrs {:value 5} :results [i32] :as %five)
  (op arith.constant :attrs {:value 10} :results [i32] :as %ten)
  (op func.call :attrs {:callee "add"} :operands [%five %ten] :results [i32] :as %result)
  (op func.return :operands [%result]))
```

**Output:** `15` (5 + 10)

## function_factorial.lisp

Multiple function definitions with function calls:

```lisp
(defn square [x:i32] i32
  (op arith.muli :operands [x x] :results [i32] :as %result)
  (op func.return :operands [%result]))

(defn add_one [x:i32] i32
  (op arith.constant :attrs {:value 1} :results [i32] :as %one)
  (op arith.addi :operands [x %one] :results [i32] :as %result)
  (op func.return :operands [%result]))

(defn main [] i32
  (op arith.constant :attrs {:value 5} :results [i32] :as %five)
  (op func.call :attrs {:callee "square"} :operands [%five] :results [i32] :as %squared)
  (op func.call :attrs {:callee "add_one"} :operands [%squared] :results [i32] :as %result)
  (op func.return :operands [%result]))
```

**Output:** `26` (5² + 1)

## Syntax Reference

**Operation Form:**
```lisp
(op <operation-name>
    :attrs {<key> <value> ...}     ; Optional attributes
    :operands [<value> ...]         ; Optional operands (SSA values)
    :results [<type> ...]           ; Optional result types
    :as <name>)                     ; Optional: bind result to name
```

**Supported Operations:**
- `arith.constant` - Create integer/float constants
- `arith.addi` - Add two integers
- `arith.subi` - Subtract two integers
- `arith.muli` - Multiply two integers
- `arith.divsi` - Signed integer division
- `arith.addf` - Add two floats
- `arith.mulf` - Multiply two floats
- `arith.cmpi` - Compare two integers (returns i1)
- `cf.br` - Unconditional branch
- `cf.cond_br` - Conditional branch (if-then-else)
- `func.call` - Call a function
- `func.return` - Return from function

**Comparison Predicates (for arith.cmpi):**
- `"eq"` - equal
- `"ne"` - not equal
- `"slt"` - signed less than
- `"sle"` - signed less than or equal
- `"sgt"` - signed greater than
- `"sge"` - signed greater than or equal
- `"ult"` - unsigned less than
- `"ule"` - unsigned less than or equal
- `"ugt"` - unsigned greater than
- `"uge"` - unsigned greater than or equal

**Block Form:**
```lisp
(block <name> [<arg-types>...]
  <operations...>)
```

**Control Flow Keywords:**
- `:dest` - Branch destination block
- `:true` - True branch destination (for cf.cond_br)
- `:false` - False branch destination (for cf.cond_br)
- `:args` - Arguments to pass to destination block

**Block Arguments:**
- `^0`, `^1`, etc. - Reference block arguments (phi nodes)

**Function Definition Form:**
```lisp
(defn <name> [<arg>:<type> ...] <return-type>
  <operations...>)
```

Example: `(defn add [x:i32 y:i32] i32 ...)`

**Supported Types:**
- `i1` - Boolean (for conditions)
- `i8`, `i16`, `i32`, `i64` - Integer types
- `f16`, `bf16`, `f32`, `f64` - Floating point types

## Writing Your Own

1. Create a `.lisp` file
2. Write operations using the syntax above
3. Run with `cargo run --release <your-file.lisp>`
4. See the MLIR, LLVM IR, and execution result!

## fibonacci_macro.lisp

Demonstrates the macro system by rewriting the fibonacci example with cleaner syntax:

```lisp
;; Define macros for common operations
(defmacro const [value result_name]
  (op arith.constant
    :attrs {:value value}
    :results [i32]
    :as result_name))

(defmacro add [a b result_name]
  (op arith.addi
    :operands [a b]
    :results [i32]
    :as result_name))

;; Use macros to simplify the fibonacci implementation
(defn fib [n:i32] i32
  (block entry []
    (const 0 %zero)
    (const 1 %one)
    (less_or_equal n %one %base_case)
    (cond_branch %base_case return_n loop_init))
  ;; ... rest of implementation
)
```

**Output:** `55` (fib(10))

## macro_simple.lisp & macro_arithmetic.lisp

Demonstrates basic macro functionality with simple examples:

```lisp
(defmacro inc [x]
  (op arith.addi
    :operands [x %one]
    :results [i32]
    :as %result))

(inc %forty_one)
```

**macro_simple.lisp Output:** `42`
**macro_arithmetic.lisp Output:** `16` ((5 + 3) * 2)

## Macro System

The macro system supports:
- `defmacro` for defining macros
- Template substitution with parameter binding
- `quote`, `quasiquote`, and `unquote` for code generation
- Recursive macro expansion
- Two-pass compilation: macro definition collection, then expansion

**Macro Syntax:**
```lisp
(defmacro <name> [<params>...] <body>)
```

Macros are expanded before code generation, allowing you to write high-level abstractions that expand to low-level MLIR operations.

## Known Limitations

- **arith.cmpf**: Floating point comparisons not yet implemented (but integer comparisons work!)

## What's Coming Next

- Special forms: `let`, `do`
- More built-in macros for common patterns
- Pattern matching in macros
- Splicing and varargs support
