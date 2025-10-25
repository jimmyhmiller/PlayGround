# Grammar Validation

This document confirms that the reader successfully parses all examples from the grammar specification.

## Test Coverage

All four examples from `docs/grammar.md` have been tested:

### ✅ Example 1: Simple Constant
```clojure
(mlir
  (operation
    (name arith.constant)
    (result-bindings [%c0])
    (result-types !i32)
    (attributes { :value (#int 42) })
    (location (#unknown))))
```

**Validation:**
- Parses correctly as nested list structure
- Identifies `mlir` as the root
- Correctly handles `result-bindings` with vector containing `%c0`
- Properly parses type expression `!i32`
- Successfully handles map with keyword `:value` and attribute expression `#int`

### ✅ Example 2: Function with Addition
```clojure
(mlir
  (operation
    (name func.func)
    (attributes {
      :sym  (#sym @add)
      :type (!function (inputs !i32 !i32) (results !i32))
      :visibility :public
    })
    (regions
      (region
        (block [^entry]
          (arguments [ [%x !i32] [%y !i32] ])
          (operation
            (name arith.addi)
            (result-bindings [%sum])
            (operands %x %y)
            (result-types !i32))
          (operation
            (name func.return)
            (operands %sum)))))))
```

**Validation:**
- Complex nested structure parses correctly
- Attributes map with 3 key-value pairs (6 elements flat)
- Block label `^entry` parsed as `block_id`
- Arguments vector contains nested vectors for arg bindings
- Multiple operations within block handled correctly
- Nested operations and regions work properly

### ✅ Example 3: Function Call with Constants
```clojure
(mlir
  (operation
    (name func.func)
    (attributes {
      :sym (#sym @main)
      :type (!function (inputs) (results !i32))
    })
    (regions
      (region
        (block [^entry]
          (arguments [])
          (operation
            (name arith.constant)
            (result-bindings [%a])
            (result-types !i32)
            (attributes { :value (#int 1) }))
          (operation
            (name arith.constant)
            (result-bindings [%b])
            (result-types !i32)
            (attributes { :value (#int 2) }))
          (operation
            (name func.call)
            (result-bindings [%r])
            (result-types !i32)
            (operands %a %b)
            (attributes { :callee (#flat-symbol @add) }))
          (operation
            (name func.return)
            (operands %r)))))))
```

**Validation:**
- Empty arguments vector `[]` handled correctly
- Multiple operations in sequence
- Symbol references `@main` and `@add` parsed properly
- Operands section with multiple value references

### ✅ Example 4: Control Flow with Multiple Blocks
```clojure
(mlir
  (operation
    (name func.func)
    (attributes {
      :sym (#sym @branchy)
      :type (!function (inputs !i1 !i32 !i32) (results !i32))
    })
    (regions
      (region
        (block [^entry]
          (arguments [ [%cond !i1] [%x !i32] [%y !i32] ])
          (operation
            (name cf.cond_br)
            (operands %cond)
            (successors
              (successor ^then (%x))
              (successor ^else (%y)))))
        (block [^then]
          (arguments [ [%t !i32] ])
          (operation (name func.return) (operands %t)))
        (block [^else]
          (arguments [ [%e !i32] ])
          (operation (name func.return) (operands %e)))))))
```

**Validation:**
- Multiple blocks in a region (3 blocks: entry, then, else)
- Successors section with multiple successor clauses
- Block references `^then`, `^else` in successors
- Operand bundles `(%x)`, `(%y)` in successors
- Different block argument patterns

## Test Results

```
Build Summary: 9/9 steps succeeded; 7/7 tests passed
```

All grammar examples parse successfully and produce the expected AST structure.

## What the Reader Validates

The reader validates:
- ✅ Correct tokenization of all special forms (`%`, `^`, `@`, `!`, `#`, `:`)
- ✅ Proper bracket matching for `()`, `[]`, `{}`
- ✅ Nested structure handling
- ✅ String literals with escaping
- ✅ Numbers (integers, floats, hex, binary)
- ✅ Comments (semicolon-prefixed)

## What the Reader Does NOT Validate

The reader intentionally does NOT validate:
- ❌ MLIR semantic rules (e.g., "operations must have a name")
- ❌ Section ordering in operations
- ❌ Required vs optional sections
- ❌ Type compatibility
- ❌ SSA form validity
- ❌ Block/value reference validity

This separation allows the reader to focus on syntax while deferring semantic validation to later compilation stages.
