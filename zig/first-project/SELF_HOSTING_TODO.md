# Self-Hosting Roadmap

## Goal
Rewrite the Lisp compiler (currently ~6000 lines of Zig) in our own Lisp language, requiring C interop and systems programming capabilities.

## Current Language Capabilities ✅

### Type System
- [x] Integers (generic + sized: U8, U16, U32, U64, I8, I16, I32, I64)
- [x] Floats (F32, F64)
- [x] Strings
- [x] Booleans
- [x] Function types `(-> [params...] return-type)`
- [x] Struct definitions
- [x] Enum definitions (basic)
- [x] Type annotations `(: Type)`

### Language Features
- [x] Arithmetic operators: `+ - * / %`
- [x] Comparison operators: `< > <= >= == !=`
- [x] Conditional expressions: `(if cond then else)`
- [x] Let bindings: `(let [x 10] body)`
- [x] Function definitions: `(def name (fn [params] body))`
- [x] Recursive functions
- [x] Higher-order functions
- [x] Struct field access: `(. struct field)`
- [x] Namespaces: `(ns name)`

### Code Generation
- [x] C backend with type checking
- [x] Namespace-based compilation (all defs in struct)
- [x] Function pointers in namespaces
- [x] Static function generation
- [x] Executable mode (`main()`)
- [x] Bundle mode (`lisp_main()`)
- [x] Forward declarations
- [x] C keyword sanitization

## Missing Features for Self-Hosting

### Critical (Required)

#### 1. Pointers & References
- [ ] Pointer types: `(Ptr T)`, `(*Ptr T)` (mutable)
- [ ] Address-of operator: `(& value)`
- [ ] Dereference operator: `(* ptr)`
- [ ] Null pointer constant
- [ ] Pointer arithmetic (optional, for array access)

**Why needed:** Memory management, data structures (linked lists, hash maps), string manipulation

#### 2. Arrays
- [ ] Fixed-size arrays: `(Array T size)`
- [ ] Array literal syntax: `[1 2 3]` or `(array 1 2 3)`
- [ ] Array indexing: `([] arr index)` or `(get arr index)`
- [ ] Array length

**Why needed:** String buffers, token arrays, parse trees

#### 3. Memory Management
- [ ] Manual allocation: `(malloc size)` or `(alloc T)`
- [ ] Manual deallocation: `(free ptr)`
- [ ] Stack allocation: `(stackalloc T size)`
- [ ] Memory copy: `(memcpy dest src size)`
- [ ] Memory set: `(memset ptr value size)`

**Why needed:** Dynamic strings, resizable arrays, hash maps

#### 4. C String Support
- [ ] C string type (null-terminated)
- [ ] String length: `(strlen str)`
- [ ] String compare: `(strcmp s1 s2)`
- [ ] String copy: `(strcpy dest src)`
- [ ] Character type and operations
- [ ] String indexing

**Why needed:** Source code processing, identifier manipulation

#### 5. I/O Operations
- [ ] File opening: `(fopen path mode)`
- [ ] File reading: `(fread buffer size count file)`
- [ ] File writing: `(fwrite buffer size count file)`
- [ ] File closing: `(fclose file)`
- [ ] Read entire file: `(read-file path)`
- [ ] Write entire file: `(write-file path content)`
- [ ] stdin/stdout/stderr access

**Why needed:** Reading source files, writing generated C code

#### 6. Error Handling
- [ ] Result type: `(Result T E)` or error unions
- [ ] Error propagation (try/catch or ? operator)
- [ ] Panic/abort: `(panic message)`
- [ ] Assert: `(assert condition message)`

**Why needed:** Compilation errors, type checking errors, file I/O errors

#### 7. Pattern Matching / Switch
- [ ] Match expressions: `(match value (pattern expr)...)`
- [ ] Enum variant matching
- [ ] Wildcard patterns

**Why needed:** AST traversal, token processing, type checking

#### 8. Hash Maps / Dictionaries
- [ ] HashMap type (can be library code once we have pointers)
- [ ] Insert, lookup, delete operations
- [ ] Iteration

**Why needed:** Symbol tables, type environments, namespace tracking

#### 9. Dynamic Arrays / Vectors
- [ ] Resizable array type (can be library once we have malloc)
- [ ] Push, pop, append operations
- [ ] Capacity management

**Why needed:** Token lists, AST node lists, expression arrays

#### 10. Loops
- [ ] While loop: `(while condition body)`
- [ ] For loop: `(for init condition step body)`
- [ ] Loop break/continue
- [ ] Foreach/iteration: `(for-each item collection body)`

**Why needed:** Iterating over tokens, processing multiple files

### Important (Highly Recommended)

#### 11. Option/Maybe Type
- [ ] `(Option T)` - represents T or nothing
- [ ] Pattern matching on Some/None

**Why needed:** Optional fields, lookup results

#### 12. Tagged Unions / Sum Types
- [ ] Better enum support with associated data
- [ ] `(Enum (Variant1 T1) (Variant2 T2)...)`

**Why needed:** AST node types, token types, value types

#### 13. Type Aliases
- [ ] `(defalias Name Type)`

**Why needed:** Cleaner code, documentation

#### 14. Mutable Variables
- [ ] `(defmut x (: Int) 0)` or `(var x 0)`
- [ ] Assignment: `(set! x value)` or `(= x value)`

**Why needed:** Loop counters, accumulation

#### 15. Multiple Return Values / Destructuring
- [ ] `(let [(a b) (get-pair)] ...)`
- [ ] Tuple support

**Why needed:** Returning multiple values from functions

#### 16. Standard Library Functions
- [ ] Math: abs, min, max, pow, etc.
- [ ] String utilities
- [ ] Collection utilities
- [ ] Type conversion functions

### Nice to Have (Can be Added Later)

#### 17. Macros (Already Supported?)
- [ ] Check if macro system is complete
- [ ] Ensure macros can generate code

#### 18. Module System
- [ ] Import/export from other files
- [ ] `(import module)` or `(require module)`

#### 19. Comments in Generated C
- [ ] Preserve line numbers for debugging
- [ ] Add source location info

#### 20. Optimizations
- [ ] Tail call optimization
- [ ] Constant folding
- [ ] Dead code elimination

## Implementation Priority

### Phase 1: Essential C Interop (Week 1-2)
1. Pointers & references
2. Arrays (fixed size)
3. C strings
4. Basic I/O (read-file, write-file)

### Phase 2: Data Structures (Week 3-4)
5. Memory management (malloc/free)
6. Dynamic arrays
7. Hash maps
8. Loops (while, for)

### Phase 3: Control Flow (Week 5)
9. Pattern matching/switch
10. Error handling
11. Mutable variables

### Phase 4: Bootstrap (Week 6-8)
12. Write lexer in Lisp
13. Write parser in Lisp
14. Write type checker in Lisp
15. Write code generator in Lisp

## Testing Strategy

Each feature should have:
1. Unit tests (like `tests/run_namespace_tests.sh`)
2. Integration test (compile a non-trivial program)
3. Documentation in CLAUDE.md

## Success Criteria

The language can self-host when we can:
1. Read a .lisp source file
2. Lex it into tokens
3. Parse tokens into AST
4. Type check the AST
5. Generate C code
6. Write C code to file
7. Compile C to executable

All done in Lisp code compiled by the current Zig implementation.
