# JRust Language Reference

JRust is a Rust-like language that compiles to JVM bytecode. The compiler is self-hosting — it is written in JRust and can compile itself.

## Quick Example

```rust
struct Point {
    x: f64,
    y: f64,
}

impl Point {
    fn new(x: f64, y: f64) -> Point {
        let p = Point { x: x, y: y };
        return p;
    }

    fn distance(self) -> f64 {
        return (self.x * self.x + self.y * self.y);
    }
}

enum Shape {
    Circle { radius: f64 },
    Rect { w: f64, h: f64 },
}

fn describe(s: Shape) -> String {
    match s {
        Shape::Circle { radius } => {
            return "circle r=" + String::from(radius);
        }
        Shape::Rect { w, h } => {
            return "rect " + String::from(w) + "x" + String::from(h);
        }
        _ => { return "unknown"; }
    }
    return "";
}

fn main() {
    let p = Point::new(3.0, 4.0);
    println("Distance: " + String::from(p.distance()));

    let shapes: Vec<Shape> = Vec::new();
    shapes.push(Shape::Circle { radius: 5.0 });
    shapes.push(Shape::Rect { w: 10.0, h: 20.0 });

    let mut i: i32 = 0;
    while i < shapes.len() {
        println(describe(shapes.get(i)));
        i = i + 1;
    }
}
```

## Types

### Primitives

| Type | Description | JVM Mapping |
|------|-------------|-------------|
| `i32` | 32-bit signed integer | `int` |
| `i64` | 64-bit signed integer | `long` |
| `f64` | 64-bit float | `double` |
| `bool` | boolean | `int` (0/1) |
| `char` | character | `char` |

### Reference Types

| Type | Description | JVM Mapping |
|------|-------------|-------------|
| `String` | immutable string | `java.lang.String` |
| `Vec<T>` | growable list | `java.util.ArrayList` |
| `Map<K, V>` | hash map | `java.util.HashMap` |
| `StringBuilder` | mutable string builder | `java.lang.StringBuilder` |

### User-Defined Types

**Structs** — product types with named fields:
```rust
struct Token {
    kind: String,
    value: String,
    line: i32,
}
```

**Enums** — sum types with optional fields per variant:
```rust
enum Expr {
    IntLit { value: i64 },
    BinaryOp { left: Expr, op: String, right: Expr },
    Ident { name: String },
    CallExpr { name: String, args: Vec<Expr> },
    NullLit,
}
```

### Generics

Generic type parameters on `Vec` and `Map`:
```rust
let names: Vec<String> = Vec::new();
let counts: Map<String, i32> = Map::new();
let nested: Vec<Vec<i32>> = Vec::new();
```

### Type Annotations

Type annotations are required on `let` bindings unless the type can be inferred from the initializer:
```rust
let x: i32 = 5;
let name = "hello";        // inferred as String
let v: Vec<i32> = Vec::new();
let mut count = 0;          // inferred as i32
```

## Expressions

### Literals

```rust
42              // i32
42L             // i64 (not supported; use let x: i64 = 42)
3.14            // f64
"hello world"   // String (supports \n \t \r \\ \" \' \0)
'a'             // char
true            // bool
false           // bool
null            // null reference
```

### Arithmetic Operators

All work on `i32`, `i64`, and `f64`:
```rust
a + b       // addition (also string concatenation)
a - b       // subtraction
a * b       // multiplication
a / b       // division
a % b       // modulus
```

### Comparison Operators

```rust
a == b      // equality (uses .equals for strings, null-safe)
a != b      // inequality
a < b       // less than
a > b       // greater than
a <= b      // less or equal
a >= b      // greater or equal
```

### Logical Operators

```rust
a && b      // logical AND (compiled as bitwise AND on bools)
a || b      // logical OR (compiled as bitwise OR on bools)
!a          // logical NOT
```

### Bitwise Operators

```rust
a | b       // bitwise OR
a & b       // bitwise AND (via &&)
```

### String Concatenation

The `+` operator concatenates when either operand is a String:
```rust
"hello " + "world"          // "hello world"
"count: " + String::from(42) // "count: 42"
"x" + "=" + String::from(x)  // chained
```

### Field Access

```rust
point.x
token.kind
self.field_name
```

### Method Calls

```rust
obj.method()
obj.method(arg1, arg2)
vec.push(42)
str.len()
```

### Static Calls

```rust
Vec::new()
Map::new()
StringBuilder::new()
Point::new(3.0, 4.0)
JRustAsm::cw_new(3)        // imported class methods
```

### Struct Initialization

```rust
Point { x: 3.0, y: 4.0 }
Token { kind: "IDENT", value: name, line: 1 }
```

### Enum Variant Initialization

```rust
Expr::IntLit { value: 42 }
Expr::NullLit               // no fields
Color::RGB { r: 255, g: 0, b: 0 }
```

### Indexing

```rust
vec.get(0)          // Vec indexing (via method)
str.char_at(0)      // String character access
```

### Type Casting

```rust
value as i64        // i32 → i64
value as f64        // i32/i64 → f64
value as i32        // f64/i64 → i32
expr as SomeType    // reference cast (CHECKCAST)
```

### Assignment

```rust
x = 10;
self.field = value;
```

### Throw

```rust
throw "error message";
```

## Control Flow

### If / Else

```rust
if condition {
    // then
}

if x > 0 {
    println("positive");
} else {
    println("non-positive");
}
```

### While Loop

```rust
let mut i: i32 = 0;
while i < 10 {
    println(String::from(i));
    i = i + 1;
}
```

### For Range Loop

Iterates from start (inclusive) to end (exclusive):
```rust
for i in 0..10 {
    println(String::from(i));
}
```

### For Each Loop

Iterates over a `Vec` using Java's Iterator protocol:
```rust
for item in my_vec {
    println(item);
}
```

### Match

Pattern matching on values, strings, integers, and enum variants:

```rust
match expr {
    Expr::IntLit { value } => {
        println("int: " + String::from(value));
    }
    Expr::Ident { name } => {
        println("ident: " + name);
    }
    _ => {
        println("other");
    }
}
```

**Pattern types:**
- `_` — wildcard, matches anything
- `42`, `"text"`, `true`, `false`, `null`, `'c'` — literal patterns
- `EnumName::Variant { field1, field2 }` — enum variant with field bindings
- `EnumName::Variant` — enum variant without fields

### Break / Continue

```rust
while true {
    if done {
        break;
    }
    if skip {
        continue;
    }
}
```

## Statements

### Let Bindings

```rust
let x = 5;                         // immutable, inferred type
let x: i32 = 5;                    // explicit type
let mut count: i32 = 0;            // mutable
let name: String = "hello";
let items: Vec<String> = Vec::new();
```

### Return

```rust
fn add(a: i32, b: i32) -> i32 {
    return a + b;
}

fn do_stuff() {
    // ...
    return;     // void return
}
```

### Expression Statements

Any expression followed by `;`:
```rust
println("hello");
vec.push(42);
self.process();
```

## Declarations

### Functions

```rust
fn name(param1: Type1, param2: Type2) -> ReturnType {
    // body
}

fn void_fn() {
    // no return type = void
}

fn main() {
    // entry point
}
```

### Structs

```rust
struct Name {
    field1: Type1,
    field2: Type2,
}
```

All fields are public.

### Impl Blocks

```rust
impl StructName {
    // Static method (no self parameter)
    fn new(x: i32) -> StructName {
        let s = StructName { field: x };
        return s;
    }

    // Instance method
    fn get_field(self) -> i32 {
        return self.field;
    }

    // Mutating method
    fn set_field(mut self, val: i32) {
        self.field = val;
    }
}
```

### Enums

```rust
enum Direction {
    North,
    South,
    East,
    West,
}

enum Option {
    Some { value: String },
    None,
}
```

Enums compile to a base class with subclasses per variant. Variant fields become public instance fields.

### Constants

```rust
const MAX_SIZE: i32 = 1024;
const PI: f64 = 3.14159;
const NAME: String = "JRust";
```

Constants become `public static final` fields on the `Main` class.

### Imports

```rust
import jrust.JRustAsm;
import java.util.ArrayList;
```

Imports map to JVM class names and enable static method calls on the imported class.

## Built-in Functions

### Output

```rust
println("hello");          // prints to stdout with newline
println(42);               // converts to string first
print("no newline");       // prints without newline
eprintln("error");         // prints to stderr
```

### Program Control

```rust
panic("something went wrong");  // throws RuntimeException
exit(1);                         // System.exit(code)
```

### File I/O

```rust
let content: String = read_file("input.txt");
write_file("output.txt", content);
```

### System Commands

```rust
let exit_code: i32 = system("ls -la");
```

### Command-Line Arguments

```rust
let count: i32 = args_len();
let first: String = args_get(0);
```

## Built-in Methods

### String

| Method | Signature | Description |
|--------|-----------|-------------|
| `.len()` | `() -> i32` | String length |
| `.char_at(i)` | `(i32) -> char` | Character at index |
| `.contains(s)` | `(String) -> bool` | Substring check |
| `.starts_with(s)` | `(String) -> bool` | Prefix check |
| `.ends_with(s)` | `(String) -> bool` | Suffix check |
| `.substring(start)` | `(i32) -> String` | Substring from index |
| `.substring(start, end)` | `(i32, i32) -> String` | Substring range |
| `.equals(s)` | `(String) -> bool` | Equality |
| `.replace(old, new)` | `(String, String) -> String` | Replace all |
| `.trim()` | `() -> String` | Trim whitespace |
| `.is_empty()` | `() -> bool` | Empty check |

### Vec\<T\>

| Method | Signature | Description |
|--------|-----------|-------------|
| `.push(val)` | `(T) -> void` | Append element |
| `.get(i)` | `(i32) -> T` | Get element (auto-unboxed) |
| `.set(i, val)` | `(i32, T) -> void` | Replace element |
| `.len()` | `() -> i32` | Number of elements |
| `.is_empty()` | `() -> bool` | Empty check |
| `.remove(i)` | `(i32) -> T` | Remove at index |
| `.contains(val)` | `(T) -> bool` | Membership check |

### Map\<K, V\>

| Method | Signature | Description |
|--------|-----------|-------------|
| `.insert(k, v)` | `(K, V) -> void` | Put key-value pair |
| `.get(k)` | `(K) -> V` | Get value for key |
| `.contains_key(k)` | `(K) -> bool` | Key existence check |
| `.get_or_default(k, d)` | `(K, V) -> V` | Get with default |
| `.len()` | `() -> i32` | Number of entries |
| `.is_empty()` | `() -> bool` | Empty check |
| `.remove(k)` | `(K) -> V` | Remove entry |

### StringBuilder

| Method | Signature | Description |
|--------|-----------|-------------|
| `.append(val)` | `(T) -> StringBuilder` | Append value |
| `.to_string()` | `() -> String` | Convert to String |
| `.len()` | `() -> i32` | Current length |

### Static Constructors

```rust
Vec::new()              // empty Vec
Map::new()              // empty Map
StringBuilder::new()    // empty StringBuilder
String::from(value)     // convert i32/i64/f64/char/bool to String
```

## Java Interop

### Importing Java Classes

```rust
import jrust.JRustAsm;
```

After importing, static methods can be called:
```rust
let handle: i32 = JRustAsm::cw_new(3);
JRustAsm::cw_visit(handle, version, access, name, superName);
```

Method descriptors for imported classes are inferred from argument types and the expected return type at the call site.

### JRustAsm API

The `JRustAsm` class provides a handle-based wrapper around the ASM bytecode library. All ASM objects (ClassWriter, MethodVisitor, Label, etc.) are stored internally and referenced by integer handles.

**ClassWriter operations:**
- `cw_new(flags) -> i32` — create a ClassWriter
- `cw_visit(h, version, access, name, superName)` — set class header
- `cw_visit_method(h, access, name, desc) -> i32` — add a method
- `cw_visit_field(h, access, name, desc) -> i32` — add a field
- `cw_end(h)` — finish class
- `cw_write(h, dir, name)` — write .class file

**MethodVisitor operations:**
- `mv_code(h)` — begin code
- `mv_insn(h, opcode)` — zero-operand instruction
- `mv_int_insn(h, opcode, operand)` — int operand instruction
- `mv_var_insn(h, opcode, slot)` — local variable instruction
- `mv_field_insn(h, opcode, owner, name, desc)` — field access
- `mv_method_insn(h, opcode, owner, name, desc, isInterface)` — method call
- `mv_jump_insn(h, opcode, label)` — conditional/unconditional jump
- `mv_label(h, label)` — place label
- `mv_ldc_str(h, value)` / `mv_ldc_int` / `mv_ldc_long` / `mv_ldc_double` — load constant
- `mv_type_insn(h, opcode, type)` — type instruction (NEW, CHECKCAST, etc.)
- `mv_maxs(h, maxStack, maxLocals)` — set max stack/locals
- `mv_end(h)` — finish method

**Label:** `label_new() -> i32` — create a new label

**Utilities:**
- `read_file(path) -> String`
- `write_file(path, content)`
- `mkdir(path)`
- `run_command(command) -> i32`
- `parse_int(s) -> i32` / `parse_long(s) -> i64` / `parse_double(s) -> f64`
- `args_len() -> i32` / `args_get(index) -> String`

## Building

```bash
# Compile the bootstrap (Java) compiler
bash build.sh

# Compile a .jrs file
java -cp "asm.jar:build" jrust.Main myfile.jrs

# Run the compiled output
java -cp "output:asm.jar:build" Main

# Self-hosting: compile compiler.jrs, then use the output to compile again
java -cp "asm.jar:build" jrust.Main compiler.jrs
cp output/*.class self/
java -cp "self:asm.jar:build" Main compiler.jrs
```

## Differences from Rust

- **No ownership/borrowing** — all objects are garbage collected
- **No lifetimes** — managed by JVM GC
- **No traits** — use impl blocks on structs
- **No pattern matching in let** — only in `match`
- **No closures/lambdas**
- **No modules** — single-file compilation
- **No tuple types**
- **No Result/Option in stdlib** — define your own enums
- **No `&` references** — all non-primitives are heap references
- **`&&`/`||` are not short-circuit** — compiled as bitwise AND/OR on booleans
- **Comments** — only `//` line comments, no `/* */` block comments
