# Project


We are building a low-level, statically typed lisp. Our lisp is a bit different. Our only special forms will be mlir. We are going to have a syntax that will be able to map one to one with mlir. Everything else is going to be built out of macros. So for example, not builtin if, instead we use scf.if and we have a macro `if` that expands into that.



* see grammar.md in docs for details on what the language looks like

# Important Zig 0.15.1 Notes

## ArrayList Usage

**CRITICAL:** In Zig 0.15.1, `ArrayList` does NOT have an `.init()` method and does NOT store the allocator as a field.

### Correct Pattern

```zig
// Initialize with empty struct literal
var list = std.ArrayList(i32){};
defer list.deinit(allocator);

// Pass allocator to EVERY method call
try list.append(allocator, 42);
try list.appendSlice(allocator, &[_]i32{1, 2, 3});
try list.resize(allocator, 10);
try list.ensureTotalCapacity(allocator, 100);

// Converting to owned slice
const owned = try list.toOwnedSlice(allocator);
defer allocator.free(owned);
```

### WRONG Patterns (DO NOT USE)

```zig
// WRONG: No .init() method exists
var list = std.ArrayList(i32).init(allocator); // COMPILE ERROR

// WRONG: No allocator field exists
var list = std.ArrayList(i32){};
list.allocator = allocator; // COMPILE ERROR

// WRONG: Methods require allocator parameter
try list.append(42); // COMPILE ERROR - missing allocator
```

### Common Operations

```zig
const allocator = std.testing.allocator;

// Create
var list = std.ArrayList(i32){};
defer list.deinit(allocator);

// Add items
try list.append(allocator, 10);
try list.appendSlice(allocator, &[_]i32{20, 30});

// Access
const len = list.items.len;
const first = list.items[0];

// Clear
list.clearRetainingCapacity(); // keeps memory allocated
list.clearAndFree(allocator);  // frees memory

// Convert to slice
const owned = try list.toOwnedSlice(allocator);
defer allocator.free(owned);
// Note: Don't call deinit after toOwnedSlice
```

See `/Users/jimmyhmiller/Documents/Code/PlayGround/zig/mlir-lisp/test/arraylist_usage_test.zig` for comprehensive examples and patterns.

# Tools



## Bug Tracker

Use this tool to record bugs discovered during development. This helps track issues that need to be addressed later. Each bug gets a unique ID (goofy animal name like "curious-elephant") for easy reference and closing.

### Tool Definition

```json
{
  "name": "bug_tracker",
  "description": "Records bugs discovered during development to BUGS.md in the project root. Each bug gets a unique goofy animal name ID. Includes AI-powered quality validation.",
  "input_schema": {
    "type": "object",
    "properties": {
      "project": {
        "type": "string",
        "description": "Project root directory path"
      },
      "title": {
        "type": "string",
        "description": "Short bug title/summary"
      },
      "description": {
        "type": "string",
        "description": "Detailed description of the bug"
      },
      "file": {
        "type": "string",
        "description": "File path where bug was found"
      },
      "context": {
        "type": "string",
        "description": "Code context like function/class/module name where bug was found"
      },
      "severity": {
        "type": "string",
        "enum": ["low", "medium", "high", "critical"],
        "description": "Bug severity level"
      },
      "tags": {
        "type": "string",
        "description": "Comma-separated tags"
      },
      "repro": {
        "type": "string",
        "description": "Minimal reproducing case or steps to reproduce"
      },
      "code_snippet": {
        "type": "string",
        "description": "Code snippet demonstrating the bug"
      },
      "metadata": {
        "type": "string",
        "description": "Additional metadata as JSON string (e.g., version, platform)"
      }
    },
    "required": ["project", "title"]
  }
}
```

### Usage

Add a bug:
```bash
bug-tracker add --title <TITLE> [OPTIONS]
```

Close a bug:
```bash
bug-tracker close <BUG_ID>
```

List bugs:
```bash
bug-tracker list
```

View a bug:
```bash
bug-tracker view <BUG_ID>
```

### Examples

**Add a comprehensive bug report:**
```bash
bug-tracker add --title "Null pointer dereference" --description "Found potential null pointer access" --file "src/main.rs" --context "authenticate()" --severity high --tags "memory,safety" --repro "Call authenticate with null user_ptr" --code-snippet "if (!user_ptr) { /* missing check */ }"
```

**Close a bug by ID:**
```bash
bug-tracker close curious-elephant
```

**View a bug by ID:**
```bash
bug-tracker view curious-elephant
```

**Enable AI quality validation:**
```bash
bug-tracker add --title "Bug title" --description "Bug details" --validate
```

The `--validate` flag triggers AI-powered quality checking to ensure bug reports contain sufficient information before recording.

# Ongoing Work

## Migration: Value (Tagged Union) → CValueLayout (C-Compatible Struct)

### Current State (2025-01-04)

We currently have **two parallel representations** for values in the codebase:

1. **`Value`** (`src/reader.zig`) - Zig tagged union
   - Used throughout parser, printer, macro expander, builder
   - Ergonomic pattern matching: `switch (value.type)`
   - Type-safe field access: `value.data.atom`, `value.data.list`
   - Variable size (~16-24 bytes for atoms)

2. **`CValueLayout`** (`src/reader/c_value_layout.zig`) - C-compatible extern struct
   - Used at JIT macro boundary
   - Fixed 56-byte size, flat memory layout
   - C ABI compatible for MLIR FFI
   - Now has ergonomic methods: `asAtom()`, `asList()`, `match()`

**Current conversion happens in**: `src/jit_macro_wrapper.zig`
- `convertValueToCValueLayout()` - recursively converts Value → CValueLayout
- `convertCValueLayoutToValue()` - recursively converts CValueLayout → Value
- Only needed at JIT macro boundary (not in hot paths)

### Why Migrate?

**Benefits of using CValueLayout everywhere:**
1. **Single source of truth** - No dual representations
2. **No conversions** - Eliminate conversion overhead at JIT boundary
3. **MLIR integration** - Direct memory layout for FFI without translation
4. **Consistency** - Same type in Zig and JIT-compiled code

**Challenges:**
- ~50+ files use `Value` extensively
- Pattern matching becomes more verbose
- Need to thread allocators for collection access
- Lose compile-time tagged union safety

### New Ergonomic API (Completed ✓)

`CValueLayout` now has inline methods making it nearly as ergonomic as tagged unions:

```zig
// Type checking
if (layout.isAtom()) { ... }
if (layout.isCollection()) { ... }
const vtype = layout.getType(); // Returns ValueType enum

// Type-safe accessors (return optionals)
if (layout.asAtom()) |atom| {
    std.debug.print("Atom: {s}\n", .{atom});
}

if (layout.asList()) |list| {
    for (list) |elem| { ... }
}

// Pattern matching via visitor
try layout.match(.{
    .onAtom = struct {
        fn call(atom: []const u8) !void { ... }
    }.call,
    .onList = struct {
        fn call(list: []*CValueLayout) !void { ... }
    }.call,
});
```

**All methods are `inline fn` - zero runtime overhead!**

### Migration Plan

#### Phase 1: Preparation (Current)
- [x] Add ergonomic methods to CValueLayout
- [x] Add comprehensive tests for new API
- [x] Document usage patterns
- [ ] Audit all Value usage sites (~50+ files)
- [ ] Create migration helper utilities

#### Phase 2: Core Types (Parser/Printer/Reader)
Files to update:
- `src/reader.zig` - Change public API to return `*CValueLayout`
- `src/parser.zig` - Update to work with CValueLayout
- `src/printer.zig` - Update to work with CValueLayout
- `src/tokenizer.zig` - May need updates depending on Value usage

Pattern to follow:
```zig
// OLD
const value = try allocator.create(Value);
value.* = Value{
    .type = .identifier,
    .data = .{ .atom = "name" },
};

// NEW
const layout = try allocator.create(CValueLayout);
layout.* = CValueLayout.empty(.identifier);
layout.data_ptr = @constCast("name".ptr);
layout.data_len = "name".len;

// OR use constructor helper (to be added)
const layout = try CValueLayout.createAtom(allocator, .identifier, "name");
```

#### Phase 3: Macro System
Files to update:
- `src/macro_expander.zig` - Update pattern matching to use `.match()` or switch
- `src/builtin_macros.zig` - Update macro implementations
- Remove conversion code from `src/jit_macro_wrapper.zig`

Key change - pattern matching:
```zig
// OLD
switch (value.type) {
    .list => {
        const list = value.data.list;
        // ...
    },
    .identifier => {
        const name = value.data.atom;
        // ...
    },
}

// NEW (Option 1: inline switch)
switch (layout.getType()) {
    .list => {
        if (layout.asList()) |list| {
            // ...
        }
    },
    .identifier => {
        if (layout.asAtom()) |name| {
            // ...
        }
    },
}

// NEW (Option 2: visitor pattern)
try layout.match(.{
    .onList = handleList,
    .onAtom = handleAtom,
});
```

#### Phase 4: Builder and Operation Types
Files to update:
- `src/builder.zig` - Update to work with CValueLayout
- `src/operation_flattener.zig` - Update Value usage
- `src/c_api_transform.zig` - May simplify with single type

#### Phase 5: Tests and Examples
- Update all test files to use CValueLayout
- Update example programs
- Run full test suite
- Performance testing to verify no regressions

#### Phase 6: Cleanup
- Remove old `Value` type from `src/reader.zig`
- Remove `valueToCLayout()` and `cLayoutToValue()` from `src/reader/c_value_layout.zig`
- Update documentation
- Remove conversion functions from `src/jit_macro_wrapper.zig`

### Helper Utilities to Add

To make migration easier, add these constructors to `CValueLayout`:

```zig
// src/reader/c_value_layout.zig

/// Create an atom value (identifier, number, string, etc.)
pub fn createAtom(
    allocator: std.mem.Allocator,
    vtype: reader.ValueType,
    atom: []const u8,
) !*CValueLayout {
    const layout = try allocator.create(CValueLayout);
    layout.* = CValueLayout.empty(vtype);
    layout.data_ptr = @constCast(atom.ptr);
    layout.data_len = atom.len;
    return layout;
}

/// Create a list from slice of CValueLayout pointers
pub fn createList(
    allocator: std.mem.Allocator,
    elements: []*CValueLayout,
) !*CValueLayout {
    const layout = try allocator.create(CValueLayout);
    layout.* = CValueLayout.empty(.list);
    layout.data_ptr = @ptrCast(elements.ptr);
    layout.data_len = elements.len;
    layout.data_capacity = elements.len;
    layout.data_elem_size = @sizeOf(*CValueLayout);
    return layout;
}

// Similar for createVector, createMap, etc.
```

### Testing Strategy

1. **Parallel implementation** - Keep both types working during migration
2. **File-by-file testing** - Migrate one file at a time, ensure tests pass
3. **Integration tests** - Run full test suite after each phase
4. **Performance benchmarks** - Verify no regressions from conversion removal

### Rollback Plan

If migration causes issues:
1. Keep git commits atomic (one file/module per commit)
2. Can revert individual files if needed
3. Both representations will work during transition
4. Conversion functions remain until Phase 6

### Estimated Effort

- **Phase 1**: ✓ Complete (2-3 hours)
- **Phase 2**: ~1 day (core types are complex)
- **Phase 3**: ~1 day (macro system is critical)
- **Phase 4**: ~0.5 days (builder is straightforward)
- **Phase 5**: ~0.5 days (update tests)
- **Phase 6**: ~0.5 days (cleanup)

**Total**: ~4 days of focused work

### Success Criteria

- [ ] All tests pass
- [ ] No Value type remains in codebase
- [ ] No conversion functions needed
- [ ] JIT macros work without conversions
- [ ] Performance is same or better
- [ ] Code remains readable and maintainable

### Open Questions

1. **PersistentVector integration** - Does it need updates to store CValueLayout efficiently?
2. **Memory management** - Are there lifetime issues with 56-byte structs vs pointers?
3. **Error handling** - Do we need better error messages for type mismatches?
4. **Builder API** - Does the builder need a fluent interface for CValueLayout construction?

### Notes

- This migration is **optional** - current dual-type system works fine
- Main benefit is eliminating conversions at JIT boundary
- Ergonomic API makes CValueLayout nearly as nice to use as Value
- Decision to proceed should be based on actual performance profiling of conversions
