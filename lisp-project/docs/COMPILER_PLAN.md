# Complete op/block Lisp Compiler Implementation Plan

## Overview
Build a full general-purpose compiler for op/block Lisp that can:
1. Read and parse any op/block program from a file
2. Translate op/block forms to MLIR using the C API (not text)
3. Lower to LLVM and JIT execute
4. Start with fibonacci as the test case

## Phase 1: S-Expression Data Structures

### File: `src/sexpr.lisp`

**Data structures needed:**
```lisp
;; Core S-expression type
(def SExpr (: Type)
  (Struct
    [tag SExprTag]
    [data (Pointer Nil)]))

(def SExprTag (: Type)
  (Enum Symbol Number String List Nil))

;; List node for linked list representation
(def ListNode (: Type)
  (Struct
    [value SExpr]
    [next (Pointer ListNode)]))

;; Symbol table entry (for intern-ing symbols)
(def SymbolEntry (: Type)
  (Struct
    [name (Pointer U8)]
    [len I32]
    [next (Pointer SymbolEntry)]))
```

**Key functions:**
- `make-symbol(name: Pointer U8) -> SExpr` - Create symbol, interned in symbol table
- `make-number(value: I64) -> SExpr` - Create number
- `make-string(str: Pointer U8) -> SExpr` - Create string
- `make-list(items: Pointer ListNode) -> SExpr` - Create list
- `sexpr-equal(a: SExpr, b: SExpr) -> Bool` - Compare s-expressions
- `sexpr-is-symbol(s: SExpr, name: Pointer U8) -> Bool` - Check if symbol matches name
- `sexpr-print(s: SExpr)` - Debug printing

## Phase 2: File Reading & Tokenization

### File: `src/reader.lisp`

**Tokenizer:**
```lisp
(def Token (: Type)
  (Struct
    [type TokenType]
    [value (Pointer U8)]
    [len I32]))

(def TokenType (: Type)
  (Enum LParen RParen LBracket RBracket Symbol Number String EOF))

(def Tokenizer (: Type)
  (Struct
    [source (Pointer U8)]
    [pos I32]
    [len I32]))
```

**Key functions:**
- `tokenizer-init(source: Pointer U8, len: I32) -> Tokenizer`
- `tokenizer-next(tok: Pointer Tokenizer) -> Token` - Get next token
- `tokenizer-peek(tok: Pointer Tokenizer) -> Token` - Peek without consuming
- `skip-whitespace(tok: Pointer Tokenizer)`
- `read-symbol(tok: Pointer Tokenizer) -> Token`
- `read-number(tok: Pointer Tokenizer) -> Token`
- `read-string(tok: Pointer Tokenizer) -> Token`

**File reading:**
- Use `fopen`, `fseek`, `ftell`, `fread` from stdlib
- Allocate buffer for entire file contents
- Pass to tokenizer

## Phase 3: S-Expression Parser

### File: `src/parser.lisp`

**Parser state:**
```lisp
(def Parser (: Type)
  (Struct
    [tokenizer Tokenizer]
    [allocator Allocator]))  ;; Simple arena allocator
```

**Key functions:**
- `parser-init(source: Pointer U8, len: I32) -> Parser`
- `parse-sexpr(p: Pointer Parser) -> SExpr` - Parse one s-expression
- `parse-list(p: Pointer Parser) -> SExpr` - Parse list/vector
- `parse-atom(p: Pointer Parser) -> SExpr` - Parse symbol/number/string

**Algorithm:**
- Recursive descent parser
- Handle `()` for lists, `[]` for vectors (treated as lists)
- Build linked list for list contents
- Return fully parsed s-expression tree

## Phase 4: MLIR Value Table

### File: `src/value_table.lisp`

**Value table (simple hash map):**
```lisp
(def ValueEntry (: Type)
  (Struct
    [name (Pointer U8)]     ;; SSA value name like "0", "1", "arg0"
    [value MlirValue]       ;; MLIR value
    [next (Pointer ValueEntry)]))

(def ValueTable (: Type)
  (Struct
    [buckets (Pointer (Pointer ValueEntry))]
    [size I32]
    [count I32]))
```

**Key functions:**
- `value-table-create(size: I32) -> Pointer ValueTable`
- `value-table-insert(tbl: Pointer ValueTable, name: Pointer U8, val: MlirValue)`
- `value-table-lookup(tbl: Pointer ValueTable, name: Pointer U8) -> MlirValue`
- `value-table-destroy(tbl: Pointer ValueTable)`

**Simple hash function:**
- Use djb2 or similar for string hashing
- Chaining for collision resolution

## Phase 5: op Form Compiler

### File: `src/op_compiler.lisp`

**op form structure:**
```
(op <name> <result-types> <operands> <attrs> <regions>)
```

**Key function:**
```lisp
(def compile-op (: (-> [SExpr MlirContext MlirLocation Pointer ValueTable I32] MlirOperation))
  (fn [op-expr ctx loc value-table result-num]
    ;; 1. Extract op name (2nd element of list)
    ;; 2. Parse result types list
    ;; 3. Resolve operands from value table
    ;; 4. Parse attributes
    ;; 5. Recursively compile regions
    ;; 6. Create MlirOperationState
    ;; 7. Add results, operands, attributes, regions
    ;; 8. Create operation
    ;; 9. Register results in value table
    ))
```

**Steps:**
1. **Extract fields:** Use `list-nth` to get each component
2. **Result types:** Parse strings like `["i32"]` into `MlirType[]`
   - Use `mlirTypeParseGet(ctx, "i32")` for each type
3. **Operands:** Parse strings like `["arg0", "0"]`
   - Look up each in value table to get `MlirValue`
4. **Attributes:** Parse map like `{"value" "1 : i32"}`
   - Convert to `MlirNamedAttribute[]`
   - Use `mlirAttributeParseGet(ctx, attrStr)`
5. **Regions:** Recursively compile each region (list of blocks)
6. **Build operation:**
   ```lisp
   let state = mlirOperationStateGet(mlirStringRefCreateFromCString(op-name), loc)
   mlirOperationStateAddResults(&state, num-results, result-types)
   mlirOperationStateAddOperands(&state, num-operands, operand-values)
   mlirOperationStateAddAttributes(&state, num-attrs, attributes)
   mlirOperationStateAddOwnedRegions(&state, num-regions, regions)
   let operation = mlirOperationCreate(&state)
   ```
7. **Register results:**
   - For each result, add to value table with name `"<result-num>"`
   - Increment result counter

## Phase 6: block Form Compiler

### File: `src/block_compiler.lisp`

**block form structure:**
```
(block <args> <ops>)
```

**Key function:**
```lisp
(def compile-block (: (-> [SExpr MlirContext MlirLocation Pointer ValueTable Pointer I32] MlirBlock))
  (fn [block-expr ctx loc value-table result-num]
    ;; 1. Extract args list
    ;; 2. Parse arg types
    ;; 3. Create MlirBlock with arguments
    ;; 4. Register block args in value table
    ;; 5. Compile each op in block
    ;; 6. Append ops to block
    ))
```

**Steps:**
1. **Parse args:** `[["arg0" "i32"] ["arg1" "i32"]]`
   - Extract names and types
   - Parse types with `mlirTypeParseGet`
2. **Create block:**
   ```lisp
   let arg-types: MlirType[] = [...]
   let arg-locs: MlirLocation[] = [...]  ;; All unknown loc
   let block = mlirBlockCreate(num-args, arg-types, arg-locs)
   ```
3. **Register args:**
   - For each arg, get `MlirValue` with `mlirBlockGetArgument(block, i)`
   - Insert into value table with arg name
4. **Compile ops:**
   - Iterate through ops list
   - Call `compile-op` for each
   - Call `mlirBlockAppendOwnedOperation(block, op)`

## Phase 7: Region Compilation

### File: `src/region_compiler.lisp`

**Region is just a list of blocks:**

**Key function:**
```lisp
(def compile-region (: (-> [SExpr MlirContext MlirLocation Pointer ValueTable Pointer I32] MlirRegion))
  (fn [region-expr ctx loc value-table result-num]
    ;; 1. Create empty region
    ;; 2. Compile each block
    ;; 3. Append blocks to region
    ))
```

**Steps:**
1. `let region = mlirRegionCreate()`
2. For each block s-expr in region list:
   - Call `compile-block`
   - Call `mlirRegionAppendOwnedBlock(region, block)`
3. Return region

## Phase 8: Module Compilation

### File: `src/module_compiler.lisp`

**Top-level compiler:**

```lisp
(def compile-module (: (-> [SExpr MlirContext MlirLocation] MlirModule))
  (fn [module-expr ctx loc]
    ;; 1. Create empty module
    ;; 2. Get module body block
    ;; 3. Create value table
    ;; 4. Compile top-level ops (usually func.func ops)
    ;; 5. Add to module body
    ))
```

**Steps:**
1. `let mod = mlirModuleCreateEmpty(loc)`
2. `let body = mlirModuleGetBody(mod)`
3. `let value-table = value-table-create(128)`
4. Extract ops from `(module <ops>...)` form
5. For each op:
   - Call `compile-op`
   - Call `mlirBlockAppendOwnedOperation(body, op)`
6. Return module

## Phase 9: Main Compiler Pipeline

### File: `src/main.lisp`

**Complete pipeline:**

```lisp
(def compile-file (: (-> [Pointer U8] I32))
  (fn [filename]
    ;; 1. Read file
    let source = read-file(filename)

    ;; 2. Parse to s-expression
    let parser = parser-init(source, strlen(source))
    let sexpr = parse-sexpr(&parser)

    ;; 3. Create MLIR context
    let ctx = mlirContextCreate()
    register-dialects(ctx)
    let loc = mlirLocationUnknownGet(ctx)

    ;; 4. Compile to MLIR module
    let mod = compile-module(sexpr, ctx, loc)

    ;; 5. Verify module
    if mlirModuleIsNull(mod):
      die("Failed to compile module")

    ;; 6. Dump for debugging
    mlirOperationDump(mlirModuleGetOperation(mod))

    ;; 7. Run lowering passes
    run-lowering-passes(ctx, mod)

    ;; 8. JIT and execute
    jit-and-run(mod, "main")

    ;; 9. Cleanup
    mlirModuleDestroy(mod)
    mlirContextDestroy(ctx)
    ))
```

## Phase 10: Pass Pipeline & JIT

### File: `src/jit.lisp`

**Lowering pipeline:**
```lisp
(def run-lowering-passes (: (-> [MlirContext MlirModule] Nil))
  (fn [ctx mod]
    let pm = mlirPassManagerCreate(ctx)
    let opm = mlirPassManagerGetAsOpPassManager(pm)

    let pipeline = "builtin.module(func.func(convert-scf-to-cf,convert-arith-to-llvm),convert-cf-to-llvm{},convert-func-to-llvm,reconcile-unrealized-casts)"

    mlirParsePassPipeline(opm, mlirStringRefCreateFromCString(pipeline), NULL, NULL)
    mlirPassManagerRunOnOp(pm, mlirModuleGetOperation(mod))
    mlirPassManagerDestroy(pm)
    ))
```

**JIT execution:**
```lisp
(def jit-and-run (: (-> [MlirModule Pointer U8] Nil))
  (fn [mod fn-name]
    let engine = mlirExecutionEngineCreate(mod, 3, 0, NULL, 0)

    let fn-ptr = mlirExecutionEngineLookup(engine, mlirStringRefCreateFromCString(fn-name))

    ;; Cast and call (assumes i32 main() for now)
    let main-fn = cast(Pointer (-> [] I32), fn-ptr)
    let result = main-fn()

    printf("Result: %d\n", result)

    mlirExecutionEngineDestroy(engine)
    ))
```

## Testing Strategy

### Test 1: Simple constant
```lisp
;; simple.lisp
(module
  (op "func.func" [] [] {"sym_name" "\"main\"" "function_type" "() -> i32"} [
    [(block [] [
      (op "arith.constant" ["i32"] [] {"value" "42 : i32"} [])
      (op "func.return" [] ["0"] {} [])
    ])]
  ]))
```

Expected: Returns 42

### Test 2: Simple arithmetic
```lisp
;; add.lisp
(module
  (op "func.func" [] [] {"sym_name" "\"main\"" "function_type" "() -> i32"} [
    [(block [] [
      (op "arith.constant" ["i32"] [] {"value" "40 : i32"} [])
      (op "arith.constant" ["i32"] [] {"value" "2 : i32"} [])
      (op "arith.addi" ["i32"] ["0" "1"] {} [])
      (op "func.return" [] ["2"] {} [])
    ])]
  ]))
```

Expected: Returns 42

### Test 3: Fibonacci (full test)
Use the `fib.lisp` we already created.

Expected: When we add a wrapper to call `fib(10)`, should return 55

## File Structure

```
lisp-project/
├── src/
│   ├── sexpr.lisp           # S-expression data structures
│   ├── reader.lisp          # File reading & tokenization
│   ├── parser.lisp          # S-expression parser
│   ├── value_table.lisp     # SSA value table
│   ├── op_compiler.lisp     # op form compiler
│   ├── block_compiler.lisp  # block form compiler
│   ├── region_compiler.lisp # region compiler
│   ├── module_compiler.lisp # top-level module compiler
│   ├── jit.lisp             # Pass pipeline & JIT
│   └── main.lisp            # Main entry point
├── tests/
│   ├── simple.lisp          # Test 1
│   ├── add.lisp             # Test 2
│   └── fib.lisp             # Test 3 (fibonacci)
└── build.sh                 # Build script

Build: ./build0/bin/lisp0 src/main.lisp -o opblock_compiler
Run: ./opblock_compiler tests/fib.lisp
```

## Implementation Notes

1. **Memory management:**
   - Use arena allocator for s-expressions (never freed during compilation)
   - MLIR owns its memory
   - Free value table at end

2. **Error handling:**
   - Check all MLIR operations for null/failure
   - Print helpful error messages with source location
   - Use `die()` helper for fatal errors

3. **String handling:**
   - Intern all symbols for fast comparison
   - Use string views where possible (pointer + length)
   - Careful with null termination for C FFI

4. **Type system:**
   - Start with i32 only
   - Expand to i1, i64, f32, f64 as needed
   - Parse MLIR type strings directly

5. **Debugging:**
   - Add `--dump-sexpr` flag to print parsed s-expressions
   - Add `--dump-mlir` flag to print MLIR before lowering
   - Add `--dump-llvm` flag to print after lowering

## Success Criteria

✓ Compile simple.lisp and get 42
✓ Compile add.lisp and get 42
✓ Compile fib.lisp and get 55 for fib(10)
✓ No hardcoded MLIR text - all built with C API
✓ General-purpose - works for any op/block program
✓ Proper error messages for malformed input

## Estimated Effort

- Phase 1-3 (Parsing): ~300 lines
- Phase 4 (Value table): ~100 lines
- Phase 5-7 (Compilation): ~400 lines
- Phase 8-10 (Pipeline): ~200 lines
- **Total: ~1000 lines of build0 Lisp code**

This is a complete, production-ready compiler for op/block Lisp!
