# Plan: MLIR AST to S-Expression Lisp Converter

## Overview
Create a new printer (`src/lisp_printer.zig`) that converts the MLIR AST to the S-expression-based Lisp format specified in the grammar.

## Architecture

### File Structure
1. **`src/lisp_printer.zig`** - Main converter module
   - `LispPrinter` struct with writer and indentation tracking
   - Functions mirroring the AST structure
   - Each function annotated with the Lisp grammar rule it implements

2. **`src/root.zig`** - Add public API function:
   - `pub fn printLisp(allocator: Allocator, module: Module) ![]const u8`
   - `pub fn formatLisp(module: Module, writer: anytype) !void`

3. **`test/lisp_printer_test.zig`** - Test conversion
   - Roundtrip tests: MLIR → AST → Lisp
   - Test each feature incrementally

## Implementation Strategy

### Phase 1: Core Structure
1. **Module wrapper** - `(mlir OPERATION*)`
2. **Operation wrapper** - `(operation (name OP_NAME) SECTION*)`
3. **Basic operations** - Constants, arithmetic ops

### Phase 2: Operation Sections
4. **Result bindings** - `(result-bindings [VALUE_ID*])`
5. **Result types** - `(result-types TYPE*)`
6. **Operands** - `(operand-uses VALUE_ID*)`
7. **Attributes** - `(attributes {KEYWORD ATTR*})`
8. **Function types** - `(result-types TYPE*)` from function type outputs

### Phase 3: Types
9. **Builtin types** - `i32`, `f64`, `index` (plain identifiers)
10. **Dialect types** - `!llvm.ptr`, `!transform.any_op` (with ! prefix)
11. **Function types** - `(!function (inputs TYPE*) (results TYPE*))`
12. **Complex types** - tensor, memref, vector, etc.

### Phase 4: Advanced Features
13. **Regions** - `(regions REGION*)`
14. **Blocks** - `(block [BLOCK_ID] (arguments [...]) OPERATION*)`
15. **Successors** - `(successor BLOCK_ID (operand-bundle)?)`
16. **Locations** - `(location ATTR)`

### Phase 5: Attributes
17. **Typed literals** - `(: 42 i32)` for integer attributes
18. **Symbol references** - `@func_name` for function names
19. **Keywords** - `:value`, `:sym_name` for attribute keys
20. **Dictionary attributes** - `{:key value :key2 value2}`

### Phase 6: Aliases
21. **Type aliases** - Convert `!alias = type` definitions
22. **Attribute aliases** - Convert `#alias = attr` definitions

## Key Mapping Challenges & Solutions

### 1. **Attribute Dictionary Conversion**
**Challenge**: MLIR uses `{key = value, key2 = value2}`, Lisp uses `{:key value :key2 value2}`

**Solution**:
- Prefix keys with `:` (convert to keyword syntax)
- Convert `=` to whitespace
- Handle typed integer literals: `42 : i32` → `(: 42 i32)`

### 2. **Function Type Representation**
**Challenge**: MLIR uses `(i32, i32) -> i32`, Lisp uses `(!function (inputs i32 i32) (results i32))`

**Solution**:
- Check if function type is in operation signature context
- In attributes, wrap with `(!function ...)`
- Split inputs/outputs into labeled sections

### 3. **Type Prefix Handling**
**Challenge**: Builtin types (i32, f64) vs dialect types (!llvm.ptr)

**Solution**:
- Builtin types: output as plain identifiers (no prefix)
- Dialect types: preserve `!` prefix
- Function types in attributes: use `!function` prefix

### 4. **Integer Attribute Values**
**Challenge**: MLIR properties use `<{value = 42 : i32}>`, need `{:value (: 42 i32)}`

**Solution**:
- Detect typed integer literals in attributes
- Wrap with `(: value type)` syntax
- Handle both typed and untyped integers

### 5. **Symbol References**
**Challenge**: Function names like `"add"` vs `@add`

**Solution**:
- In `:sym_name` attributes: output as `@symbol`
- In `:callee` attributes: output as `@function_name`
- Detect string attributes that should be symbols

### 6. **Region Entry Blocks**
**Challenge**: MLIR has unlabeled entry blocks, Lisp needs block structure

**Solution**:
- Entry block operations go in first `(block [label] (arguments [...]) ops)`
- If no label exists, can omit the label binding `(block (arguments [...]) ops)`
- Or generate implicit label for clarity

### 7. **Successor Operand Bundles**
**Challenge**: `^bb1(%arg0, %arg1)` vs `(successor ^bb1 (%arg0 %arg1))`

**Solution**:
- Detect block arguments in successors
- Output as `(successor BLOCK_ID (operand-bundle VALUES))`

### 8. **Preserving Source Information**
**Challenge**: Attributes are stored as raw strings in AST

**Solution**:
- Parse attribute raw strings to detect patterns:
  - Integer literals with types
  - String values
  - Symbol references
  - Nested structures
- Convert to appropriate Lisp syntax

## Incremental Testing Strategy

1. **Start simple**: `%0 = "arith.constant"() <{value = 42 : i32}> : () -> i32`
   - Test operation wrapper
   - Test result bindings
   - Test attributes with typed literals
   - Test function types

2. **Add complexity**: Arithmetic operations with operands
   - Test operand-uses section
   - Test multiple operations in module

3. **Control flow**: Branches and blocks
   - Test regions
   - Test blocks with labels and arguments
   - Test successors

4. **Nested regions**: scf.if with then/else
   - Test multiple regions
   - Test nested operations

5. **Aliases**: Type and attribute definitions
   - Test top-level alias definitions

## Potential Difficulties

### Critical Issues

#### 1. **Attribute parsing ambiguity**
**Problem**: Raw strings like `"42 : i32"` need semantic interpretation

**Details**:
- Current AST stores attribute values as raw strings
- Need to re-parse attribute strings to understand structure
- Must handle all MLIR attribute syntaxes (integers, floats, arrays, nested dicts, etc.)

**Risk**: May not handle all MLIR attribute syntaxes correctly

**Mitigation**:
- Start with common patterns (typed integers, strings, symbols)
- Add parsing as needed for more complex attributes
- Test against actual MLIR examples from test suite

#### 2. **Symbol vs String distinction**
**Problem**: How to know when a string should be `@symbol`?

**Details**:
- Some string attributes represent symbols (function names, etc.)
- Others are literal strings
- Context determines which is which

**Risk**: May miss some symbol references or incorrectly convert strings

**Mitigation**:
- Use heuristics based on attribute key (`:sym_name`, `:callee` → symbols)
- Document known symbol attribute keys
- Allow for manual overrides if needed

#### 3. **Function type context**
**Problem**: When is it `!function` vs plain signature?

**Details**:
- In operation signatures: plain `() -> i32`
- In attributes: `(!function (inputs) (results i32))`
- Need context-aware printing

**Risk**: May use wrong format in some contexts

**Mitigation**:
- Track context in printer (operation signature vs attribute)
- Add explicit parameter to type printing functions
- Test both contexts thoroughly

### Medium Issues

#### 4. **Dialect attribute bodies**
**Problem**: Opaque `#llvm<...>` content

**Details**:
- Currently stored as raw strings
- May contain complex nested structures
- Dialect-specific syntax

**Risk**: May not parse correctly for all dialects

**Mitigation**:
- Preserve as-is for now (output raw content)
- Can add dialect-specific parsing later if needed

#### 5. **Memory layout attributes**
**Problem**: memref layouts, tensor encodings

**Details**:
- Currently simplified/skipped in parser
- Complex syntax with affine maps, etc.

**Risk**: May lose information or produce incorrect output

**Mitigation**:
- Handle gracefully (output what we have)
- May need to extend parser first for full support
- Document limitations

#### 6. **Location information**
**Problem**: Currently stored as raw string

**Details**:
- Location syntax: `loc("file.mlir":10:5)`
- May have fused locations, name locations, etc.

**Risk**: May not format correctly in all cases

**Mitigation**:
- Output as opaque string for now
- Can parse location syntax later if needed

### Minor Issues

#### 7. **Indentation/formatting**
**Problem**: Making S-expressions readable

**Details**:
- Need to decide on pretty-printing strategy
- Balance between compact and readable
- Deep nesting can be hard to read

**Risk**: Output may be hard to read for complex IR

**Mitigation**:
- Use consistent indentation (2 spaces)
- Add newlines strategically
- Test with complex examples and adjust

#### 8. **Integer overflow in attributes**
**Problem**: i64 values in attributes

**Details**:
- Large integer values
- Need proper conversion without loss

**Risk**: May lose precision or overflow

**Mitigation**:
- Use i64 in AST (already done)
- Print full precision
- Test with large values

## Success Criteria

1. **Simple constants** roundtrip correctly
   - `%0 = "arith.constant"() <{value = 42 : i32}> : () -> i32`
   - Converts to valid Lisp with typed literal

2. **Arithmetic operations** with multiple operands work
   - Operations like `arith.addi` with operands
   - Operand-uses section populated correctly

3. **Control flow** with blocks and successors converts
   - Labeled blocks with arguments
   - Successor lists with operand bundles

4. **Nested regions** (scf.if) produce correct structure
   - Multiple regions in single operation
   - Nested operations within regions

5. **Attributes** convert properly:
   - Typed integers: `(: 42 i32)`
   - Symbols: `@func_name`
   - Keywords: `:value`
   - Dictionary syntax: `{:key value}`

6. **Function types** in attributes use `!function` form
   - `:function_type` attributes
   - Proper inputs/results sections

7. **All test examples** produce valid Lisp output
   - Every `.mlir` file in test_data/examples/
   - Output is valid according to Lisp grammar

## Example Conversions

### Example 1: Simple Constant

**MLIR**:
```mlir
%0 = "arith.constant"() <{value = 42 : i32}> : () -> i32
```

**Lisp**:
```lisp
(operation
  (name arith.constant)
  (result-bindings [%0])
  (result-types i32)
  (attributes {:value (: 42 i32)}))
```

### Example 2: Arithmetic Operation

**MLIR**:
```mlir
%2 = "arith.addi"(%0, %1) : (i32, i32) -> i32
```

**Lisp**:
```lisp
(operation
  (name arith.addi)
  (result-bindings [%2])
  (result-types i32)
  (operand-uses %0 %1))
```

### Example 3: Control Flow

**MLIR**:
```mlir
"test.cond_br"(%2)[^bb1, ^bb2] : (i1) -> ()
```

**Lisp**:
```lisp
(operation
  (name test.cond_br)
  (operand-uses %2)
  (successors
    (successor ^bb1)
    (successor ^bb2)))
```

### Example 4: Nested Region

**MLIR**:
```mlir
%1 = "scf.if"(%0) ({
  %3 = "arith.constant"() <{value = 42 : i32}> : () -> i32
  "scf.yield"(%3) : (i32) -> ()
}, {
  %2 = "arith.constant"() <{value = 0 : i32}> : () -> i32
  "scf.yield"(%2) : (i32) -> ()
}) : (i1) -> i32
```

**Lisp**:
```lisp
(operation
  (name scf.if)
  (result-bindings [%1])
  (result-types i32)
  (operand-uses %0)
  (regions
    (region
      (block
        (arguments [])
        (operation
          (name arith.constant)
          (result-bindings [%3])
          (result-types i32)
          (attributes {:value (: 42 i32)}))
        (operation
          (name scf.yield)
          (operand-uses %3))))
    (region
      (block
        (arguments [])
        (operation
          (name arith.constant)
          (result-bindings [%2])
          (result-types i32)
          (attributes {:value (: 0 i32)}))
        (operation
          (name scf.yield)
          (operand-uses %2))))))
```

## Next Steps After Approval

1. Create `src/lisp_printer.zig` with basic structure
2. Implement Phase 1-2 (core structure + sections)
3. Write tests for simple constants
4. Iterate through phases 3-6
5. Add comprehensive tests for each feature
6. Handle edge cases and refine attribute conversion
7. Document any limitations or unsupported features
