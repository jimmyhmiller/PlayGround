# Reader Implementation

This document describes the Clojure-style reader implementation for the lisp-project.

## Overview

The reader is implemented in three main modules:

1. **data.lisp** - Immutable data structures
2. **tokenizer.lisp** - Lexical analysis (breaking text into tokens)
3. **reader.lisp** - Parsing (converting tokens into data structures)

## Data Structures (src/data.lisp)

The reader supports the following immutable data structures:

### Value Types

All values are represented by a `Value` struct with a tag indicating the type:

- `TAG_NIL` (0) - nil value
- `TAG_NUMBER` (1) - Integer numbers
- `TAG_STRING` (2) - String literals
- `TAG_SYMBOL` (3) - Symbols (identifiers)
- `TAG_LIST` (4) - Linked lists (cons cells)
- `TAG_VECTOR` (5) - Vectors (dynamic arrays)
- `TAG_MAP` (6) - Maps (key-value pairs)

### Constructors

```lisp
(make-nil)                          ; Create nil
(make-number 42)                    ; Create number
(make-string "hello")               ; Create string
(make-symbol "foo")                 ; Create symbol
(make-empty-list)                   ; Create empty list
(cons head tail)                    ; Add to list
(make-empty-vector)                 ; Create empty vector
(vector-conj vec elem)              ; Add to vector
(make-empty-map)                    ; Create empty map
(map-assoc map key val)             ; Add to map
```

### Immutability

All data structures are immutable. Operations that "modify" structures actually create new copies:

- `cons` creates a new list with an element added to the front
- `vector-conj` creates a new vector with an element appended
- `map-assoc` creates a new map with a key-value pair added

Immutability is achieved through copying. While not efficient, it satisfies the requirement of maintaining immutable data structures.

## Tokenizer (src/tokenizer.lisp)

The tokenizer breaks input text into a stream of tokens.

### Token Types

- `TOK_LPAREN` - `(`
- `TOK_RPAREN` - `)`
- `TOK_LBRACKET` - `[`
- `TOK_RBRACKET` - `]`
- `TOK_LBRACE` - `{`
- `TOK_RBRACE` - `}`
- `TOK_NUMBER` - Integer literals
- `TOK_STRING` - String literals (enclosed in `"`)
- `TOK_SYMBOL` - Symbols/identifiers
- `TOK_EOF` - End of input

### Features

- Whitespace handling (spaces, tabs, newlines)
- Comment support (`;` to end of line)
- Symbol characters: letters, digits, `-`, `_`, `+`, `*`, `/`, `<`, `>`, `=`, `!`, `?`, `.`

### Usage

```lisp
(let [tokens (tokenize "(+ 1 2)")]
  ; Process tokens...
)
```

## Reader (src/reader.lisp)

The reader converts a stream of tokens into data structures.

### Main Functions

```lisp
;; Read a single form
(read tokens)

;; Read all forms into a vector
(read-all tokens)
```

### Supported Forms

1. **Numbers**: `42`, `123`
2. **Strings**: `"hello world"`
3. **Symbols**: `foo`, `bar`, `+`, `-`
4. **Lists**: `(+ 1 2)`, `(def x 10)`
5. **Vectors**: `[1 2 3]`, `[:a :b :c]`
6. **Maps**: `{:name "John" :age 30}`
7. **Nested structures**: `(def data {:users [{:name "Alice"} {:name "Bob"}]})`

## Example Usage

```lisp
;; Complete example of reading Clojure-style code

(let [input "(def x [1 2 3])"]
  (let [tokens (tokenize input)]
    (let [value (read tokens)]
      ; value is now a List containing:
      ; - Symbol "def"
      ; - Symbol "x"
      ; - Vector [1, 2, 3]
      value)))
```

## Integration Note

Due to lisp0's lack of a built-in module system, using the reader requires concatenating the source files in order:

1. `src/data.lisp` (data structures)
2. `src/tokenizer.lisp` (tokenization)
3. `src/reader.lisp` (reading/parsing)
4. Your application code

Alternatively, each file can be compiled separately and linked together if the compiler supports that workflow.

## Testing

See `tests/reader_test.lisp` for example usage and test cases.

## Implementation Notes

### Simplicity Over Performance

The implementation prioritizes simplicity and correctness over performance:

- Lists use simple linked lists (cons cells)
- Vectors copy all elements on every conj operation
- Maps use linear arrays with no hashing
- No memory pooling or optimization

This is intentional for the bootstrap phase. Performance can be improved later.

### Memory Management

All allocations use `malloc`. There is currently no garbage collection or explicit memory freeing. For the bootstrap phase, this is acceptable. A proper memory management system can be added later.

### Type System

The reader is compatible with lisp0's type system:
- Explicit type annotations on all definitions
- Typed function signatures
- Struct-based data representation

## Future Enhancements

Potential improvements for later phases:

1. **Better error messages** - Line/column tracking in tokens
2. **More data types** - Keywords (`:keyword`), characters, booleans
3. **Reader macros** - Quote (`'`), syntax quote (`` ` ``), unquote (`~`)
4. **Metadata** - `^{:doc "..."} symbol`
5. **Performance** - Persistent data structures with structural sharing
6. **Garbage collection** - Automatic memory management
7. **Unicode support** - UTF-8 string handling

## Design Decisions

### Why Immutable?

Immutable data structures provide:
- Predictable behavior
- Easy reasoning about code
- No aliasing bugs
- Foundation for functional programming

### Why Simple Copying?

For the bootstrap phase, copying is:
- Easy to implement
- Easy to verify correctness
- Sufficient for small data structures
- A good foundation for later optimization

The goal is to get working infrastructure first, then optimize.
