# Fix Symbol Stripping in Type Checker

## Problem Summary

**13 special forms** strip their operator/keyword symbol during type checking, forcing the code generator to use fragile heuristics. This creates maintenance nightmares and potential bugs.

### Root Cause

In `type_checker.zig`, many `synthesizeTyped*` functions return typed lists without their leading operator symbol. For example:
- Input: `(if (< x 10) 1 2)` - 4 elements
- After type checking: `[cond_typed, then_typed, else_typed]` - 3 elements, **no "if"**

This forces the code generator (`simple_c_compiler.zig:1804+`) to guess what operation a list represents using:
- Element count patterns
- Type signature matching
- Operator exclusion lists

**Example of the problem:**
```zig
// From simple_c_compiler.zig:2079-2108
// Check for if expression (represented as 3-element list without 'if' symbol)
// Pattern: [cond, then, else] where cond has boolean type
// IMPORTANT: Make sure first element is NOT an operator symbol (like <, >, =, etc.)
if (l.elements.len == 3) {
    const cond_type = l.elements[0].getType();
    const is_operator_form = l.elements[0].* == .symbol and blk: {
        const op_name = l.elements[0].symbol.name;
        break :blk std.mem.eql(u8, op_name, "<") or
            std.mem.eql(u8, op_name, ">") or
            // ... many more checks
    };
    if (cond_type == .bool and !is_operator_form) {
        // This MIGHT be an if expression...
    }
}
```

## Changes Required

### Phase 1: Fix Type Checker (`type_checker.zig`)

All `synthesizeTyped*` functions must include the operator/keyword symbol as the first element.

#### ❌ BROKEN Forms (need fixing):

#### 1. **`synthesizeTypedIf`** (line ~1950)
**Current:**
```zig
storage[0] = cond_typed;
storage[1] = then_typed;
storage[2] = else_typed;
result.* = TypedValue{ .list = .{
    .elements = storage[0..3],
    .type = result_type,
} };
```

**Fixed:**
```zig
const if_symbol = try self.allocator.create(TypedValue);
if_symbol.* = TypedValue{ .symbol = .{ .name = "if", .type = result_type } };
storage[0] = if_symbol;
storage[1] = cond_typed;
storage[2] = then_typed;
storage[3] = else_typed;
result.* = TypedValue{ .list = .{
    .elements = storage[0..4],  // Now 4 elements!
    .type = result_type,
} };
```

#### 2. **`synthesizeTypedWhile`** (line ~1986)
**Current:**
```zig
storage[0] = cond_typed;
// body expressions at storage[1..]
```

**Fixed:**
```zig
const while_symbol = try self.allocator.create(TypedValue);
while_symbol.* = TypedValue{ .symbol = .{ .name = "while", .type = Type.void } };
storage[0] = while_symbol;
storage[1] = cond_typed;
// body expressions at storage[2..]
// Update elem_count to account for symbol
```

#### 3. **`synthesizeTypedCFor`** (line ~2074)
**Current:**
```zig
storage[0] = init_typed;
storage[1] = cond_typed;
storage[2] = step_typed;
// body expressions follow
```

**Fixed:**
```zig
const cfor_symbol = try self.allocator.create(TypedValue);
cfor_symbol.* = TypedValue{ .symbol = .{ .name = "c-for", .type = Type.void } };
storage[0] = cfor_symbol;
storage[1] = init_typed;
storage[2] = cond_typed;
storage[3] = step_typed;
// body expressions follow
// Update elem_count indices
```

#### 4. **`synthesizeTypedSet`** (line ~2103)
**Current:**
```zig
storage[0] = value_typed;
result.* = TypedValue{ .list = .{
    .elements = storage[0..1],
    .type = Type.void,
} };
```

**Fixed:**
```zig
const set_symbol = try self.allocator.create(TypedValue);
set_symbol.* = TypedValue{ .symbol = .{ .name = "set!", .type = Type.void } };
const var_symbol = try self.allocator.create(TypedValue);
var_symbol.* = TypedValue{ .symbol = .{ .name = var_name, .type = var_type } };
storage[0] = set_symbol;
storage[1] = var_symbol;
storage[2] = value_typed;
result.* = TypedValue{ .list = .{
    .elements = storage[0..3],
    .type = Type.void,
} };
```

#### 5. **`synthesizeTypedCStr`** (line ~1802)
**Current:**
```zig
storage[0] = arg_typed;
result.* = TypedValue{ .list = .{
    .elements = storage[0..1],
    .type = Type.c_string,
} };
```

**Fixed:**
```zig
const cstr_symbol = try self.allocator.create(TypedValue);
cstr_symbol.* = TypedValue{ .symbol = .{ .name = "c-str", .type = Type.c_string } };
storage[0] = cstr_symbol;
storage[1] = arg_typed;
result.* = TypedValue{ .list = .{
    .elements = storage[0..2],
    .type = Type.c_string,
} };
```

#### 6. **`synthesizeTypedPrintf`** (line ~1858)
**Current:**
```zig
// Args stored starting at storage[0]
storage[arg_count] = arg_typed;
result.* = TypedValue{ .list = .{
    .elements = storage[0..arg_count],
    .type = Type.i32,
} };
```

**Fixed:**
```zig
const printf_symbol = try self.allocator.create(TypedValue);
printf_symbol.* = TypedValue{ .symbol = .{ .name = "printf", .type = Type.i32 } };

// Type check all arguments into temp array, then shift
var temp_storage: [64]*TypedValue = undefined;
var arg_count: usize = 0;
// ... type check args into temp_storage[0..arg_count]

storage[0] = printf_symbol;
var i: usize = 0;
while (i < arg_count) : (i += 1) {
    storage[i + 1] = temp_storage[i];
}

result.* = TypedValue{ .list = .{
    .elements = storage[0..arg_count+1],
    .type = Type.i32,
} };
```

#### 7. **`synthesizeTypedBitwise`** (line ~1880)
**Current (unary):**
```zig
storage[0] = operand_typed;
result.* = TypedValue{ .list = .{
    .elements = storage[0..1],
    .type = operand_type,
} };
```

**Fixed (unary):**
```zig
const op_symbol = try self.allocator.create(TypedValue);
op_symbol.* = TypedValue{ .symbol = .{ .name = op, .type = operand_type } };
storage[0] = op_symbol;
storage[1] = operand_typed;
result.* = TypedValue{ .list = .{
    .elements = storage[0..2],
    .type = operand_type,
} };
```

**Current (binary):**
```zig
storage[0] = left_typed;
storage[1] = right_typed;
result.* = TypedValue{ .list = .{
    .elements = storage[0..2],
    .type = result_type,
} };
```

**Fixed (binary):**
```zig
const op_symbol = try self.allocator.create(TypedValue);
op_symbol.* = TypedValue{ .symbol = .{ .name = op, .type = result_type } };
storage[0] = op_symbol;
storage[1] = left_typed;
storage[2] = right_typed;
result.* = TypedValue{ .list = .{
    .elements = storage[0..3],
    .type = result_type,
} };
```

#### 8. **`allocate`** in `synthesizeTyped` (line ~3153)
**Current:**
```zig
typed_elements[0] = typed_value;
result.* = TypedValue{ .list = .{
    .elements = typed_elements[0..1],
    .type = Type{ .pointer = ptr_type },
} };
```

**Fixed:**
```zig
const alloc_symbol = try self.allocator.create(TypedValue);
alloc_symbol.* = TypedValue{ .symbol = .{ .name = "allocate", .type = Type{ .pointer = ptr_type } } };
const type_marker = try self.allocator.create(TypedValue);
type_marker.* = TypedValue{ .type_value = .{ .value_type = pointee_type, .type = Type.type_type } };

typed_elements[0] = alloc_symbol;
typed_elements[1] = type_marker;
if (has_value) {
    typed_elements[2] = typed_value;
    elements = typed_elements[0..3];
} else {
    elements = typed_elements[0..2];
}

result.* = TypedValue{ .list = .{
    .elements = elements,
    .type = Type{ .pointer = ptr_type },
} };
```

#### 9. **`dereference`** in `synthesizeTyped` (line ~3225)
**Current:**
```zig
typed_elements[0] = typed_ptr;
result.* = TypedValue{ .list = .{
    .elements = typed_elements[0..1],
    .type = pointee_type,
} };
```

**Fixed:**
```zig
const deref_symbol = try self.allocator.create(TypedValue);
deref_symbol.* = TypedValue{ .symbol = .{ .name = "dereference", .type = pointee_type } };
typed_elements[0] = deref_symbol;
typed_elements[1] = typed_ptr;
result.* = TypedValue{ .list = .{
    .elements = typed_elements[0..2],
    .type = pointee_type,
} };
```

#### 10. **`pointer-write!`** in `synthesizeTyped` (line ~3255)
**Current:**
```zig
typed_elements[0] = typed_ptr;
typed_elements[1] = typed_value;
result.* = TypedValue{ .list = .{
    .elements = typed_elements[0..2],
    .type = Type.nil,
} };
```

**Fixed:**
```zig
const write_symbol = try self.allocator.create(TypedValue);
write_symbol.* = TypedValue{ .symbol = .{ .name = "pointer-write!", .type = Type.nil } };
typed_elements[0] = write_symbol;
typed_elements[1] = typed_ptr;
typed_elements[2] = typed_value;
result.* = TypedValue{ .list = .{
    .elements = typed_elements[0..3],
    .type = Type.nil,
} };
```

#### 11. **`address-of`** in `synthesizeTyped` (line ~3281)
**Current:**
```zig
typed_elements[0] = try self.allocator.create(TypedValue);
typed_elements[0].* = TypedValue{ .symbol = .{
    .name = var_name,
    .type = Type{ .pointer = ptr_type },
} };
result.* = TypedValue{ .list = .{
    .elements = typed_elements[0..1],
    .type = Type{ .pointer = ptr_type },
} };
```

**Fixed:**
```zig
const addr_symbol = try self.allocator.create(TypedValue);
addr_symbol.* = TypedValue{ .symbol = .{ .name = "address-of", .type = Type{ .pointer = ptr_type } } };
const var_symbol = try self.allocator.create(TypedValue);
var_symbol.* = TypedValue{ .symbol = .{ .name = var_name, .type = var_type } };
typed_elements[0] = addr_symbol;
typed_elements[1] = var_symbol;
result.* = TypedValue{ .list = .{
    .elements = typed_elements[0..2],
    .type = Type{ .pointer = ptr_type },
} };
```

#### 12. **`uninitialized`** in `synthesizeTyped` (line ~3171)
**Current:**
```zig
result.* = TypedValue{ .list = .{
    .elements = typed_elements[0..0],
    .type = value_type,
} };
```

**Fixed:**
```zig
const uninit_symbol = try self.allocator.create(TypedValue);
uninit_symbol.* = TypedValue{ .symbol = .{ .name = "uninitialized", .type = value_type } };
const type_marker = try self.allocator.create(TypedValue);
type_marker.* = TypedValue{ .type_value = .{ .value_type = value_type, .type = Type.type_type } };
typed_elements[0] = uninit_symbol;
typed_elements[1] = type_marker;
result.* = TypedValue{ .list = .{
    .elements = typed_elements[0..2],
    .type = value_type,
} };
```

#### 13. **`pointer-field-read` and `pointer-field-write!`**
**TODO:** Verify these are handled correctly in `synthesizeTyped` around line 3291+

---

### Phase 2: Fix Code Generator (`simple_c_compiler.zig`)

#### Remove Heuristic Detection Code

**DELETE** the entire if-expression heuristic section (line ~2079-2108):
```zig
// Check for if expression (represented as 3-element list without 'if' symbol)
// Pattern: [cond, then, else] where cond has boolean type
// IMPORTANT: Make sure first element is NOT an operator symbol (like <, >, =, etc.)
if (l.elements.len == 3) {
    const cond_type = l.elements[0].getType();
    const is_operator_form = l.elements[0].* == .symbol and blk: {
        const op_name = l.elements[0].symbol.name;
        break :blk std.mem.eql(u8, op_name, "<") or
            std.mem.eql(u8, op_name, ">") or
            std.mem.eql(u8, op_name, "<=") or
            std.mem.eql(u8, op_name, ">=") or
            std.mem.eql(u8, op_name, "=") or
            std.mem.eql(u8, op_name, "!=") or
            std.mem.eql(u8, op_name, "and") or
            std.mem.eql(u8, op_name, "or");
    };
    if (cond_type == .bool and !is_operator_form) {
        // This is an if expression (cond then else)
        // Emit as C ternary: (cond ? then : else)
        try writer.print("(", .{});
        try self.writeExpressionTyped(writer, l.elements[0], ns_ctx, includes);
        try writer.print(" ? ", .{});
        try self.writeExpressionTyped(writer, l.elements[1], ns_ctx, includes);
        try writer.print(" : ", .{});
        try self.writeExpressionTyped(writer, l.elements[2], ns_ctx, includes);
        try writer.print(")", .{});
        return;
    }
}
```

#### Update All Handlers to Expect Symbol at Index 0

For each special form handler in `writeExpressionTyped`, update to expect the symbol at `l.elements[0]` and adjust subsequent indices by +1.

**Example: `if` expression**
```zig
// NEW: Handle if by checking first element
if (l.elements.len == 4 and l.elements[0].* == .symbol and
    std.mem.eql(u8, l.elements[0].symbol.name, "if")) {
    // elements: [if, cond, then, else]
    try writer.print("(", .{});
    try self.writeExpressionTyped(writer, l.elements[1], ns_ctx, includes);  // cond
    try writer.print(" ? ", .{});
    try self.writeExpressionTyped(writer, l.elements[2], ns_ctx, includes);  // then
    try writer.print(" : ", .{});
    try self.writeExpressionTyped(writer, l.elements[3], ns_ctx, includes);  // else
    try writer.print(")", .{});
    return;
}
```

**Forms to update:**
1. `if` - adjust from 3 to 4 elements, indices +1
2. `while` - adjust indices +1
3. `c-for` - adjust indices +1
4. `set!` - adjust from 1 to 3 elements, indices +1
5. `c-str` - adjust from 1 to 2 elements, indices +1
6. `printf` - adjust indices +1
7. Bitwise ops - adjust indices +1
8. `allocate` - adjust indices +1
9. `dereference` - adjust from 1 to 2 elements, indices +1
10. `pointer-write!` - adjust from 2 to 3 elements, indices +1
11. `address-of` - adjust from 1 to 2 elements, indices +1
12. `uninitialized` - adjust from 0 to 2 elements, indices +1

---

### Phase 3: Update Tests

All tests that check typed AST structure must be updated to expect the leading symbol.

**Files to update:**
- `src/type_checker_comprehensive_tests.zig`
- `src/backend_tests.zig`
- `src/array_tests.zig`
- `src/struct_field_access_tests.zig`
- `src/showcase_test.zig`

**Example update:**
```zig
// OLD expectation:
// if expression has 3 elements: [cond, then, else]

// NEW expectation:
// if expression has 4 elements: [if_symbol, cond, then, else]
```

---

## Benefits After Refactoring

1. **Eliminates 400+ lines** of heuristic detection code
2. **Self-documenting** typed AST - you can see what operation each list represents
3. **Enables dispatch table** refactoring (future optimization)
4. **Prevents bugs** from ambiguous representations
5. **Easier debugging** - typed AST is readable
6. **Simpler code generator** - no guessing needed

---

## Checklist

### Phase 1: Type Checker
- [ ] Fix `synthesizeTypedIf`
- [ ] Fix `synthesizeTypedWhile`
- [ ] Fix `synthesizeTypedCFor`
- [ ] Fix `synthesizeTypedSet`
- [ ] Fix `synthesizeTypedCStr`
- [ ] Fix `synthesizeTypedPrintf`
- [ ] Fix `synthesizeTypedBitwise` (unary)
- [ ] Fix `synthesizeTypedBitwise` (binary)
- [ ] Fix `allocate` handler
- [ ] Fix `dereference` handler
- [ ] Fix `pointer-write!` handler
- [ ] Fix `address-of` handler
- [ ] Fix `uninitialized` handler
- [ ] Verify `pointer-field-read` and `pointer-field-write!`

### Phase 2: Code Generator
- [ ] Remove if-expression heuristic code
- [ ] Update `if` handler
- [ ] Update `while` handler
- [ ] Update `c-for` handler
- [ ] Update `set!` handler
- [ ] Update `c-str` handler
- [ ] Update `printf` handler
- [ ] Update bitwise operators handlers
- [ ] Update `allocate` handler
- [ ] Update `dereference` handler
- [ ] Update `pointer-write!` handler
- [ ] Update `address-of` handler
- [ ] Update `uninitialized` handler
- [ ] Update `pointer-field-read` handler
- [ ] Update `pointer-field-write!` handler

### Phase 3: Tests
- [ ] Update type checker tests
- [ ] Update backend tests
- [ ] Update array tests
- [ ] Update struct field access tests
- [ ] Update showcase test
- [ ] Run full test suite: `zig test src/test_all.zig`

---

## Notes

- **Order matters**: Fix type checker first, then code generator, then tests
- **Test incrementally**: Run tests after each fix to catch issues early
- **Check element counts**: Many bugs will be off-by-one errors due to the added symbol
- **Watch for storage array sizes**: Some functions may need larger storage arrays to accommodate the extra symbol element
