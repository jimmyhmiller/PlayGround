# Op Macro: Optional Operands Feature

## Summary

The `op` macro now supports optional operands vectors. Previously, operands were required to be in a vector `[...]`. Now, if there are no operands, you can omit the vector entirely.

## Changes Made

### Code Changes

1. **src/builtin_macros.zig:682-697** - Modified operand extraction logic
   - Operands vector is now optional
   - If the element at `operands_index` is not a vector, it's treated as the start of regions
   - `regions_start` is now dynamic based on whether operands were found

2. **src/builtin_macros.zig:595-604** - Updated documentation
   - Added examples showing all supported forms
   - Clarified that operands are optional

3. **test/op_macro_test.zig:336-434** - Added new tests
   - Test for operation with no operands (no vector at all)
   - Test for operation with attributes but no operands

### Supported Forms

```lisp
;; 1. With binding, type, and operands
(op %N (: index) (memref.dim [%B %c1]))

;; 2. With type and operands
(op (: index) (memref.dim [%B %c1]))

;; 3. With operands only
(op (memref.store [%value %C %i %j]))

;; 4. With attributes and operands
(op (memref.dim {attrs} [%B %c1]))

;; 5. With attributes, no operands (NEW!)
(op (memref.dim {attrs}))

;; 6. No attributes, no operands (NEW!)
(op (memref.dim))

;; 7. With operands and regions
(op %result (: i32) (scf.if [%cond]
                            (region ...)
                            (region ...)))

;; 8. With regions but no operands (NEW!)
(op %result (: i32) (test.op
                      (region ...)))
```

## How It Works

The `op` macro parses arguments in this order:

1. **Optional binding**: `%name`
2. **Optional type annotation**: `(: type)`
3. **Operation call**: `(operation-name ...)`
   - Position 0: operation name (required)
   - Position 1 (optional): attribute map `{...}`
   - Position 2 or 1 (optional): operands vector `[...]`
   - Remaining positions: regions

**Key behavior**: If the element at the "operands position" is not a vector, it's treated as the start of regions. This makes the operands vector truly optional.

## Examples

See `examples/op_macro_optional_operands.lisp` for comprehensive examples showing all use cases.

## Testing

All tests pass, including new tests for:
- Operations with no operands (no vector)
- Operations with attributes but no operands

Run tests with:
```bash
zig build test --summary all
```

## Backward Compatibility

✅ Fully backward compatible - all existing code with operands vectors continues to work exactly as before.
