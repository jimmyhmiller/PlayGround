# Zig 0.15 ArrayList API Changes

## Problem

In Zig 0.15, the ArrayList API changed significantly:

### Old API (Zig 0.11-0.13)
```zig
var list = std.ArrayList(T).init(allocator);
try list.append(item);
list.deinit();
```

### New API (Zig 0.15)
```zig
var list = std.ArrayList(T){}; // Empty initialization
try list.append(allocator, item); // Pass allocator to append
list.deinit(allocator); // Pass allocator to deinit
```

## Required Changes

### 1. Initialization
- **OLD**: `std.ArrayList(T).init(allocator)`
- **NEW**: `std.ArrayList(T){}`

### 2. Append
- **OLD**: `list.append(item)`
- **NEW**: `list.append(allocator, item)`

### 3. Deinit
- **OLD**: `list.deinit()`
- **NEW**: `list.deinit(allocator)`

## Files That Need Updates

All files using ArrayList:
- `src/reader_types.zig` ✅ FIXED
- `src/tokenizer.zig` ✅ FIXED
- `src/reader.zig` ✅ FIXED
- `src/ast.zig` ✅ FIXED
- `src/parser.zig` ⚠️ NEEDS FIXING
- `src/mlir_integration.zig` ⚠️ NEEDS FIXING
- `src/main.zig` ⚠️ NEEDS FIXING

## Pattern Matching for Fixes

### Find ArrayList init
```bash
grep -rn "ArrayList.*\.init(allocator)" src/
```

### Find append calls
```bash
grep -rn "\.append(" src/ | grep -v "allocator"
```

### Find deinit calls
```bash
grep -rn "\.deinit()" src/
```

## Common Patterns to Fix

### Pattern 1: List/Vector initialization in struct
```zig
// OLD
.list = std.ArrayList(*Value).init(allocator)

// NEW
.list = std.ArrayList(*Value){}
```

### Pattern 2: Temporary ArrayList
```zig
// OLD
var items = std.ArrayList(T).init(allocator);
defer items.deinit();
try items.append(item);

// NEW
var items = std.ArrayList(T){};
defer items.deinit(allocator);
try items.append(allocator, item);
```

### Pattern 3: Error cleanup
```zig
// OLD
var list = std.ArrayList(T).init(allocator);
errdefer list.deinit();

// NEW
var list = std.ArrayList(T){};
errdefer list.deinit(allocator);
```

### Pattern 4: Test cleanup
```zig
// OLD
defer tokens.deinit();

// NEW
defer tokens.deinit(allocator);
```

## Migration Strategy

1. **Search for `.init(allocator)`** → Replace with `{}`
2. **Search for `.append(`** → Add allocator as first parameter
3. **Search for `.deinit()`** → Add allocator as parameter

## Example Migration

Before:
```zig
pub fn createList(allocator: std.mem.Allocator) !*Value {
    const val = try allocator.create(Value);
    val.* = .{
        .type = .List,
        .data = .{ .list = std.ArrayList(*Value).init(allocator) },
        .allocator = allocator,
    };
    return val;
}

pub fn deinit(self: *Value) void {
    switch (self.type) {
        .List => {
            for (self.data.list.items) |item| {
                item.deinit();
            }
            self.data.list.deinit();
        },
        ...
    }
}

pub fn listAppend(self: *Value, item: *Value) !void {
    try self.data.list.append(item);
}
```

After:
```zig
pub fn createList(allocator: std.mem.Allocator) !*Value {
    const val = try allocator.create(Value);
    val.* = .{
        .type = .List,
        .data = .{ .list = std.ArrayList(*Value){} },
        .allocator = allocator,
    };
    return val;
}

pub fn deinit(self: *Value) void {
    switch (self.type) {
        .List => {
            for (self.data.list.items) |item| {
                item.deinit();
            }
            self.data.list.deinit(self.allocator); // Pass allocator!
        },
        ...
    }
}

pub fn listAppend(self: *Value, item: *Value) !void {
    try self.data.list.append(self.allocator, item); // Pass allocator!
}
```

## Testing After Changes

```bash
# Test just compilation
zig build

# Run all tests
zig build test

# Run REPL
zig build run
```

## What's Left to Fix

Run these commands to find remaining issues:

```bash
# Find remaining .init calls
grep -rn "ArrayList.*\.init" src/

# Find appends without allocator
grep -rn "\.append(" src/parser.zig src/mlir_integration.zig src/main.zig | grep -v "allocator"

# Find deinits without allocator
grep -rn "\.deinit()" src/parser.zig src/mlir_integration.zig src/main.zig
```

## Script to Help

Create `fix_arraylist.sh`:
```bash
#!/bin/bash
# Find all ArrayList.init calls
echo "=== ArrayList.init calls to fix ==="
grep -rn "ArrayList.*\.init(allocator)" src/

echo ""
echo "=== .deinit() calls to fix ==="
grep -rn "\.deinit()" src/

echo ""
echo "=== .append calls (check if they have allocator) ==="
grep -rn "\.append(" src/
```

Then: `chmod +x fix_arraylist.sh && ./fix_arraylist.sh`
