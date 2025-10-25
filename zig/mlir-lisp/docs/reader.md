# Reader

The reader transforms tokens from the tokenizer into structured values and collections.

## Usage

```zig
const std = @import("std");
const mlir_lisp = @import("mlir_lisp");

pub fn example() !void {
    const allocator = std.heap.page_allocator;

    const source = "(operation (name arith.constant))";

    // Create tokenizer
    var tok = mlir_lisp.Tokenizer.init(allocator, source);

    // Create reader
    var reader = try mlir_lisp.Reader.init(allocator, &tok);

    // Read one value
    var value = try reader.read();
    defer {
        value.deinit(allocator);
        allocator.destroy(value);
    }

    // value is now a structured Value
    std.debug.assert(value.type == .list);
}
```

## Value Types

The reader produces `Value` structures with the following types:

### Atoms
- `identifier` - Plain identifiers (e.g., `foo`, `arith.constant`)
- `number` - Numbers (e.g., `42`, `3.14`, `0xFF`)
- `string` - String literals (e.g., `"hello"`)
- `value_id` - SSA value IDs (e.g., `%x`, `%result`)
- `block_id` - Block labels (e.g., `^entry`, `^loop`)
- `symbol` - Symbol table names (e.g., `@main`, `@add`)
- `keyword` - Keywords (e.g., `:value`, `:type`)
- `true_lit` - Boolean true
- `false_lit` - Boolean false

### Collections
- `list` - S-expressions `(...)` - general lists
- `vector` - Square brackets `[...]` - used for bindings
- `map` - Curly braces `{...}` - stored as flat key-value pairs

### Special Markers
- `type_expr` - Type expressions `!type` (e.g., `!i32`)
- `attr_expr` - Attribute expressions `#attr` (e.g., `#(int 42)`)

## Value Structure

```zig
pub const Value = struct {
    type: ValueType,
    data: union {
        atom: []const u8,                           // For all atoms
        list: PersistentVector(*Value),             // For lists (...)
        vector: PersistentVector(*Value),           // For vectors [...]
        map: PersistentVector(*Value),              // For maps {...}
        type_expr: *Value,                          // For !...
        attr_expr: *Value,                          // For #...
    },
};
```

## Memory Management

Values must be explicitly freed using `deinit()`:

```zig
var value = try reader.read();
defer {
    value.deinit(allocator);
    allocator.destroy(value);
}
```

The `deinit()` method recursively frees all child values in collections.

## Reading Multiple Values

Use `readAll()` to read all values until EOF:

```zig
var values = try reader.readAll();
defer {
    const slice = values.slice();
    for (slice) |v| {
        v.deinit(allocator);
        allocator.destroy(v);
    }
    values.deinit();
}
```

## Error Handling

The reader can return these errors:

- `UnexpectedEOF` - Reached end of input unexpectedly
- `UnexpectedClosingDelimiter` - Closing delimiter without opening
- `UnexpectedDot` - Dot token in invalid position
- `UnterminatedList` - List/vector/map not closed
- Plus all tokenizer errors and allocation errors

## Examples

### Simple List
```clojure
(1 2 3)
```
Results in a `list` value containing three `number` values.

### MLIR Operation
```clojure
(operation
  (name arith.constant)
  (result-bindings [%c0])
  (result-types !i32)
  (attributes { :value #(int 42) }))
```
Results in a nested structure:
- Outer `list` with 5 elements
- `name` clause is a `list` with `identifier` "name" and `identifier` "arith.constant"
- `result-bindings` has a `vector` containing `value_id` "%c0"
- `result-types` has a `type_expr` wrapping `identifier` "i32"
- `attributes` has a `map` with keyword `:value` and `attr_expr` wrapping a list

## Design Notes

The reader does **not** validate MLIR structure. It only:
- Parses tokens into values
- Validates bracket matching
- Handles special markers (`!`, `#`, `%`, etc.)

Semantic validation (e.g., "operations must have a name") is left to later compilation stages.
