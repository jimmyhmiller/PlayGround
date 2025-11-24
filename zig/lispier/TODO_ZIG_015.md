# Zig 0.15 Migration TODO

## Status: Partially Complete

### ✅ Files Fully Fixed

1. **src/reader_types.zig** - All ArrayList usage updated
2. **src/tokenizer.zig** - All ArrayList usage updated
3. **src/reader.zig** - All ArrayList usage updated
4. **src/ast.zig** - All ArrayList usage updated

### ⚠️ Files Needing Fixes

#### src/parser.zig (Top Priority)

**Changes Needed:**
- Line 197: `var arr = std.ArrayList(AttributeValue).init(self.allocator);` → `{}`
- Multiple `.append()` calls need allocator parameter (lines: 121, 131, 166, 174, 200, 241, 250, 255, 308, 339, 350, 369, 375, 404, 409, 454, 465)
- Multiple `.deinit()` calls in tests need allocator parameter

**Quick Fix Command:**
```bash
# Manually edit src/parser.zig and:
# 1. Change ArrayList.init(allocator) → ArrayList{}
# 2. Change .append(X) → .append(allocator, X)
# 3. Change .deinit() → .deinit(allocator)
```

#### src/mlir_integration.zig

**Changes Needed:**
- Check for ArrayList usage (dialect/operation lists)
- Update any .append() and .deinit() calls

#### src/main.zig

**Changes Needed:**
- Check CompileResult ArrayList members
- Update any test cleanup code

#### tests/integration_test.zig

**Changes Needed:**
- Update all `.deinit()` calls to pass allocator

## How to Fix

### Manual Approach (Recommended)

For each file:

1. Search for `ArrayList.*init(allocator)` → Replace with `{}`
2. Search for `.append(` → Add `allocator,` as first parameter
3. Search for `.deinit()` → Add `allocator` as parameter

### Example

Before:
```zig
try op.regions.append(region_node.data.region);
```

After:
```zig
try op.regions.append(self.allocator, region_node.data.region);
```

### Verification

After fixing each file:
```bash
zig build
```

## When Done

Once all files are fixed:

```bash
# Full build
zig build

# Run tests
zig build test

# Try the REPL
zig build run
```

## Reference

See **CLAUDE.md** for complete migration guide with examples.
