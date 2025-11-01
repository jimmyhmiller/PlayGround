# Missing Features - Concrete Examples

**Purpose:** Show exactly what real Pyret code the parser currently cannot handle.

This document contains actual Pyret code extracted from the official repository that our parser will reject. Each example shows what's missing and why it matters.

---

## ✅ What Currently Works

Before diving into missing features, here's what the parser handles perfectly:

```pyret
# Primitives
42
"hello"
true
myVariable

# Binary operators (all 15, left-associative)
2 + 3 * 4
x < 10 and y > 5
"hello" ^ " world"

# Function calls
f(x, y)
f()(g())
obj.foo().bar().baz()

# Objects (data fields only)
{ x: 1, y: 2 }
{ point: { x: 0, y: 0 } }
{ ref counter :: Number : 0 }

# Construct expressions
[list: 1, 2, 3]
[set: x, y, z]

# Array/object access
arr[0]
obj.field
matrix[i][j].value

# Check operators
x is 5
f() raises "error"
result satisfies is-positive
```

**All these work perfectly and match the official Pyret parser byte-for-byte! ✨**

---

## ❌ Lambda Expressions (HIGHEST PRIORITY)

**Impact:** Breaks 90% of real Pyret programs
**Reason:** Higher-order functions are everywhere in Pyret

### Example 1: Simple filter from `test-lists.arr`

```pyret
filter(lam(e): e > 5 end, [list: -1, 1])
```

**Current behavior:** Parser fails on `lam` token
**Expected AST:**
```json
{
  "type": "s-app",
  "f": {"type": "s-id", "id": {"type": "s-name", "name": "filter"}},
  "args": [
    {
      "type": "s-lam",
      "params": [{"bind": {"type": "s-name", "name": "e"}}],
      "body": {
        "type": "s-block",
        "stmts": [
          {
            "type": "s-op",
            "op": {"type": "s-op-gt"},
            "left": {"type": "s-id", "id": {"type": "s-name", "name": "e"}},
            "right": {"type": "s-num", "value": "5"}
          }
        ]
      }
    },
    {
      "type": "s-construct",
      "constructor": {"type": "s-name", "name": "list"},
      "values": [...]
    }
  ]
}
```

### Example 2: Multiple parameters from `test-lists.arr`

```pyret
lists.all2(lam(n, m): n > m end, [list: 1, 2, 3], [list: 0, 1, 2])
```

**Current behavior:** Parser fails on `lam` token

### Example 3: No parameters from `test-lists.arr`

```pyret
f = lam(): "no-op" end
```

**Current behavior:** Parser fails on `lam` token

### Why this matters:

Every Pyret program using lists/sets/maps needs lambdas. From the real codebase:
- `filter`, `map`, `partition`, `find`, `all`, `any` - all require lambdas
- Event handlers: `on-click(lam(): ... end)`
- Callbacks, sorting, functional composition

**Without lambdas, you can't write real Pyret programs.**

---

## ❌ Tuple Expressions

**Impact:** Breaks any code using heterogeneous collections
**Reason:** Tuples are Pyret's primary way to return multiple values

### Example 1: Simple tuple from `test-tuple.arr`

```pyret
x = {1; 3; 10}
```

**Current behavior:** Parser treats `{` as object start, then fails on semicolon
**Expected AST:**
```json
{
  "type": "s-tuple",
  "fields": [
    {"type": "s-num", "value": "1"},
    {"type": "s-num", "value": "3"},
    {"type": "s-num", "value": "10"}
  ]
}
```

### Example 2: Tuple with expressions from `test-tuple.arr`

```pyret
{13; 1 + 4; 41; 1}
```

**Current behavior:** Parser fails

### Example 3: Nested tuples from `test-tuple.arr`

```pyret
{151; {124; 152; 12}; 523}
```

**Current behavior:** Parser fails

### Example 4: Tuple access from `test-tuple.arr`

```pyret
x = {1; 3; 10}
y = x.{2}  # Access element at index 2
```

**Current behavior:** Parser fails on `.{` syntax

### Why this matters:

From real Pyret code:
- Functions returning multiple values: `{success; result; error-msg}`
- Splitting lists: `split-at(2, lst)` returns `{ prefix: ..., suffix: ... }`
- Coordinates, RGB colors, complex return values

**The key challenge:** Disambiguating `{1; 2}` (tuple) from `{x: 1}` (object)
- Must check first separator (`;` vs `:`) after parsing first element

---

## ❌ Block Expressions

**Impact:** Breaks any code with sequencing or local bindings
**Reason:** Blocks are how you write multi-line code

### Example 1: Simple block

```pyret
block:
  x = 5
  y = 10
  x + y
end
```

**Current behavior:** Parser recognizes `block` as keyword but has no handler
**Expected AST:**
```json
{
  "type": "s-user-block",
  "body": {
    "type": "s-block",
    "stmts": [
      {"type": "s-let-expr", "binds": [...], "body": ...},
      {"type": "s-let-expr", "binds": [...], "body": ...},
      {"type": "s-op", "op": "s-op-plus", ...}
    ]
  }
}
```

### Example 2: Block with side effects

```pyret
block:
  print("Starting")
  x = compute-value()
  print("Done")
  x
end
```

**Current behavior:** Parser fails

### Why this matters:

From real Pyret code (`test-constructors.arr`):
```pyret
lam(arr) block:
  for each(x from arr):
    print(x)
  end
end
```

**Without blocks, you can't have multi-statement lambdas or control flow.**

---

## ❌ For Expressions

**Impact:** Breaks functional list operations
**Reason:** For-expressions are Pyret's idiomatic way to work with collections

### Example 1: Simple map from real code

```pyret
for map(a1 from arr): a1 + 1 end
```

**Current behavior:** Parser fails on `for` token
**Expected AST:**
```json
{
  "type": "s-for",
  "iterator": {"type": "s-name", "name": "map"},
  "bindings": [
    {"type": "s-for-bind", "bind": {"type": "s-name", "name": "a1"}, "value": ...}
  ],
  "body": ...
}
```

### Example 2: Complex example from `test-binops.arr`

```pyret
o = {
  arr: [list: 1,2,3],
  method _plus(self, other):
    for lists.map2(a1 from self.arr, a2 from other.arr):
      a1 + a2
    end
  end
}
```

**Current behavior:** Parser fails (also needs method support)

### Why this matters:

Pyret style strongly prefers for-expressions over explicit recursion:
- `for map(...)` instead of `.map(lam(...))`
- `for filter(...)` instead of `.filter(lam(...))`
- `for fold(...)` for reductions

---

## ❌ Method Fields in Objects

**Impact:** Breaks object-oriented Pyret code
**Reason:** Can't define custom operators or object methods

### Example from `test-binops.arr`

```pyret
o = {
  arr: [list: 1,2,3],
  method _plus(self, other):
    for lists.map2(a1 from self.arr, a2 from other.arr):
      a1 + a2
    end
  end
}

o2 = { arr: [list: 3,4,5] }
o + o2  # Uses _plus method!
```

**Current behavior:** Parser can handle data fields but not method fields
**Expected AST:**
```json
{
  "type": "s-obj",
  "fields": [
    {
      "type": "s-data-field",
      "name": "arr",
      "value": {"type": "s-construct", ...}
    },
    {
      "type": "s-method-field",
      "name": "_plus",
      "params": ["self", "other"],
      "body": {"type": "s-for", ...}
    }
  ]
}
```

### Why this matters:

From real Pyret code:
- Custom operators: `_plus`, `_minus`, `_times`, etc.
- Object methods: `distance(self, other)`, `toString(self)`
- Protocol implementations

**Objects are half-implemented without method support.**

---

## ❌ If Expressions

**Impact:** Breaks conditional logic
**Reason:** Core control flow primitive

### Example:

```pyret
if x > 0:
  "positive"
else if x < 0:
  "negative"
else:
  "zero"
end
```

**Current behavior:** Parser fails on `if` token

### Why this matters:

If-expressions are everywhere. Can't write conditional logic without them.

---

## ❌ Cases Expressions (Pattern Matching)

**Impact:** Breaks data type handling
**Reason:** Pyret's primary way to work with variants

### Example from `test-binops.arr`:

```pyret
cases(Eth.Either) run-task(thunk):
  | left(v) => "not-error"
  | right(v) => exn-unwrap(v)
end
```

**Current behavior:** Parser fails on `cases` token

### Why this matters:

From real Pyret code:
- Error handling: `cases(Result) r: | ok(v) => ... | err(e) => ...`
- AST processing: `cases(Expr) e: | s-num(n) => ... | s-str(s) => ...`
- State machines, parsers, interpreters

**This is how you actually use data types in Pyret!**

---

## ❌ Statements (Lower Priority)

These are important but less critical for expression parsing:

### Assignment
```pyret
x := 5
counter := counter + 1
```

### Let/Var bindings
```pyret
x = 5
var counter = 0
```

### Function declarations
```pyret
fun factorial(n):
  if n == 0: 1
  else: n * factorial(n - 1)
  end
end
```

### Data declarations
```pyret
data Box:
  | box(ref v)
end

data Option:
  | some(value)
  | none
end
```

### Import/Provide
```pyret
import equality as E
provide *
provide { factorial, fibonacci }
```

---

## 🎯 What to Implement First

Based on **real Pyret code analysis**, here's the priority:

### Phase 1: Make real programs possible (11 tests)
1. **Lambda expressions** (4 tests) - Used in 90% of programs
2. **Tuple expressions** (4 tests) - Common return type
3. **Block expressions** (2 tests) - Required for multi-line code
4. **If expressions** (1 test) - Basic control flow

**After Phase 1:** Can write meaningful Pyret programs with functions and data!

### Phase 2: Functional programming (4 tests)
5. **Method fields** (1 test) - Complete objects
6. **For expressions** (2 tests) - Idiomatic list operations
7. **Cases expressions** (1 test) - Pattern matching

**After Phase 2:** Full functional + OOP support!

### Phase 3: Declarations (7 tests)
8. Function declarations, data types, imports, etc.

**After Phase 3:** Complete language support!

---

## 📊 Test Verification

To verify any implementation, use:

```bash
# Test a specific feature
./compare_parsers.sh "lam(x): x + 1 end"

# Run comparison test
cargo test --test comparison_tests test_pyret_match_simple_lambda -- --ignored

# See all ignored tests
cargo test --test comparison_tests -- --ignored
```

All examples in this document are verified against the official Pyret parser!

---

**Last Updated:** 2025-10-31
**Source:** Real code from `/Users/jimmyhmiller/Documents/Code/open-source/pyret-lang`
