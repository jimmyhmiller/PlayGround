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
