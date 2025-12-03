# Fork Analysis: Adding ^ Metadata Support to clojure-reader

## Executive Summary

**YES, we can fork `clojure-reader` and add `^` metadata support.** This would allow us to use proper `^:dynamic` syntax instead of the earmuff convention.

**Effort Level**: Medium (3-5 days)
**Complexity**: Moderate - requires parser changes and new AST variant
**Benefits**: 100% syntax compatibility with Clojure

## Current State of clojure-reader

**Repository**: https://github.com/Grinkers/clojure-reader
**Language**: Rust
**Purpose**: Parse EDN (Extensible Data Notation) and Clojure code
**Status**: Active, well-maintained

### What's Missing

The `clojure-reader` crate currently supports:
- ✅ All EDN types (vectors, maps, sets, lists, etc.)
- ✅ Tagged literals (`#inst`, `#uuid`, etc.)
- ✅ Comments (`;`)
- ✅ Discard reader (`#_`)
- ✅ Character literals
- ✅ Keywords, symbols, strings, numbers

**Missing**:
- ❌ Metadata reader macro (`^`)
- ❌ `Edn::Meta` or `Edn::WithMeta` variant
- ❌ Metadata merging for chained metadata

## How Clojure's Metadata Reader Works

Based on [Clojure's official documentation](https://clojure.org/reference/reader):

### Basic Syntax
```clojure
^{:a 1 :b 2} [1 2 3]  ; Full map syntax
```

### Shorthand Forms

1. **Keyword shorthand**: `^:dynamic` → `^{:dynamic true}`
2. **Symbol/String shorthand**: `^String` → `^{:tag String}`
3. **Vector shorthand**: `^[String]` → `^{:param-tags [String]}`

### Chaining
```clojure
^:foo ^{:bar false} [1 2]  ; Merges to {:foo true, :bar false}
```

Metadata is **merged from right to left**.

### What Needs Metadata

Only certain objects can have metadata:
- Symbols
- Lists
- Vectors
- Sets
- Maps
- Tagged literals (if they return IMeta)

## Implementation Plan

### Step 1: Add `Edn::Meta` Variant

**File**: `/tmp/clojure-reader/src/edn.rs`

Add new variant to the `Edn` enum (line 25):

```rust
#[derive(Debug, Clone, Eq, PartialEq, PartialOrd, Ord, Hash)]
#[non_exhaustive]
pub enum Edn<'e> {
  Vector(Vec<Edn<'e>>),
  Set(BTreeSet<Edn<'e>>),
  Map(BTreeMap<Edn<'e>, Edn<'e>>),
  List(Vec<Edn<'e>>),
  Key(&'e str),
  Symbol(&'e str),
  Str(&'e str),
  Int(i64),
  Tagged(&'e str, Box<Edn<'e>>),

  // NEW: Metadata wrapper
  Meta(BTreeMap<Edn<'e>, Edn<'e>>, Box<Edn<'e>>),

  #[cfg(feature = "floats")]
  Double(OrderedFloat<f64>),
  // ... rest of variants
}
```

**Rationale**: Using `BTreeMap<Edn, Edn>` allows any EDN value as metadata keys/values, matching Clojure semantics.

### Step 2: Add Metadata Parser

**File**: `/tmp/clojure-reader/src/parse.rs`

Modify `parse_internal` to handle `^` (around line 168-218):

```rust
fn parse_internal<'e>(walker: &mut Walker, slice: &'e str) -> Result<Option<Edn<'e>>, Error> {
  walker.nibble_whitespace(slice);
  while let Some(next) = walker.peek_next(slice) {
    let column_start = walker.column;
    let ptr_start = walker.ptr;
    let line_start = walker.line;
    if let Some(ret) = match next {
      '\\' => /* ... existing char handling ... */,
      '\"' => /* ... existing string handling ... */,
      ';' => /* ... existing comment handling ... */,
      '[' => /* ... existing vector handling ... */,
      '(' => /* ... existing list handling ... */,
      '{' => /* ... existing map handling ... */,
      '#' => /* ... existing tagged/set/discard handling ... */,

      // NEW: Handle metadata reader
      '^' => {
        return Ok(Some(parse_metadata(walker, slice)?));
      }

      _ => /* ... existing literal handling ... */,
    } { /* ... */ }
  }
  Ok(None)
}
```

### Step 3: Implement `parse_metadata` Function

**File**: `/tmp/clojure-reader/src/parse.rs`

Add new function (insert after `parse_tag`):

```rust
#[inline]
fn parse_metadata<'e>(walker: &mut Walker, slice: &'e str) -> Result<Edn<'e>, Error> {
  let _ = walker.nibble_next(slice); // Consume the leading '^' char

  // Parse the metadata form
  let meta_form = match parse_internal(walker, slice)? {
    Some(edn) => edn,
    None => {
      return Err(Error {
        code: Code::UnexpectedEOF,
        line: Some(walker.line),
        column: Some(walker.column),
        ptr: Some(walker.ptr),
      });
    }
  };

  // Convert shorthand forms to maps
  let meta_map = match meta_form {
    // Keyword shorthand: ^:dynamic → {:dynamic true}
    Edn::Key(k) => {
      let mut map = BTreeMap::new();
      map.insert(Edn::Key(k), Edn::Bool(true));
      map
    }

    // Symbol/String shorthand: ^String → {:tag String}
    Edn::Symbol(s) => {
      let mut map = BTreeMap::new();
      map.insert(Edn::Key("tag"), Edn::Symbol(s));
      map
    }
    Edn::Str(s) => {
      let mut map = BTreeMap::new();
      map.insert(Edn::Key("tag"), Edn::Str(s));
      map
    }

    // Vector shorthand: ^[String] → {:param-tags [String]}
    Edn::Vector(v) => {
      let mut map = BTreeMap::new();
      map.insert(Edn::Key("param-tags"), Edn::Vector(v));
      map
    }

    // Full map syntax: ^{:a 1} → {:a 1}
    Edn::Map(m) => m,

    _ => {
      return Err(Error {
        code: Code::InvalidMetadata,  // Need to add this error code
        line: Some(walker.line),
        column: Some(walker.column),
        ptr: Some(walker.ptr),
      });
    }
  };

  // Parse the next form (the one to attach metadata to)
  let target = match parse_internal(walker, slice)? {
    Some(edn) => edn,
    None => {
      return Err(Error {
        code: Code::UnexpectedEOF,
        line: Some(walker.line),
        column: Some(walker.column),
        ptr: Some(walker.ptr),
      });
    }
  };

  // Handle chaining: if target already has metadata, merge
  let final_form = match target {
    Edn::Meta(existing_meta, inner) => {
      // Merge metadata (right to left in Clojure)
      let mut merged = existing_meta;
      for (k, v) in meta_map {
        merged.insert(k, v);
      }
      Edn::Meta(merged, inner)
    }
    _ => Edn::Meta(meta_map, Box::new(target)),
  };

  Ok(final_form)
}
```

### Step 4: Add Error Code

**File**: `/tmp/clojure-reader/src/error.rs`

Add new error variant:

```rust
pub enum Code {
  // ... existing codes
  InvalidMetadata,
}
```

### Step 5: Update Display Implementation

**File**: `/tmp/clojure-reader/src/edn.rs`

Add Display support for `Edn::Meta` (around line 163-237):

```rust
impl fmt::Display for Edn<'_> {
  fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
    match self {
      // ... existing cases

      Self::Meta(meta, value) => {
        write!(f, "^")?;
        // Try to use shorthand forms when possible
        if meta.len() == 1 {
          if let Some((k, v)) = meta.iter().next() {
            match (k, v) {
              // Keyword shorthand
              (Edn::Key(key), Edn::Bool(true)) => {
                write!(f, ":{key} {value}")
              }
              // Tag shorthand
              (Edn::Key("tag"), Edn::Symbol(s)) => {
                write!(f, "{s} {value}")
              }
              // Otherwise use full map
              _ => {
                write!(f, "{{")?;
                for (k, v) in meta {
                  write!(f, "{k} {v}")?;
                }
                write!(f, "}} {value}")
              }
            }
          } else {
            write!(f, "{value}")
          }
        } else {
          // Multiple entries, use map syntax
          write!(f, "{{")?;
          let mut it = meta.iter().peekable();
          while let Some((k, v)) = it.next() {
            if it.peek().is_some() {
              write!(f, "{k} {v}, ")?;
            } else {
              write!(f, "{k} {v}")?;
            }
          }
          write!(f, "}} {value}")
        }
      }

      // ... rest of cases
    }
  }
}
```

### Step 6: Add Tests

**File**: `/tmp/clojure-reader/tests/metadata.rs` (new file)

```rust
use clojure_reader::edn::{read_string, Edn};
use std::collections::BTreeMap;

#[test]
fn test_keyword_shorthand() {
    let result = read_string("^:dynamic x").unwrap();
    match result {
        Edn::Meta(meta, value) => {
            assert_eq!(*value, Edn::Symbol("x"));
            assert_eq!(meta.get(&Edn::Key("dynamic")), Some(&Edn::Bool(true)));
        }
        _ => panic!("Expected Meta variant"),
    }
}

#[test]
fn test_map_syntax() {
    let result = read_string("^{:a 1 :b 2} [1 2 3]").unwrap();
    match result {
        Edn::Meta(meta, value) => {
            assert!(matches!(*value, Edn::Vector(_)));
            assert_eq!(meta.get(&Edn::Key("a")), Some(&Edn::Int(1)));
            assert_eq!(meta.get(&Edn::Key("b")), Some(&Edn::Int(2)));
        }
        _ => panic!("Expected Meta variant"),
    }
}

#[test]
fn test_chained_metadata() {
    let result = read_string("^:foo ^{:bar false} [1 2]").unwrap();
    match result {
        Edn::Meta(meta, value) => {
            assert!(matches!(*value, Edn::Vector(_)));
            assert_eq!(meta.get(&Edn::Key("foo")), Some(&Edn::Bool(true)));
            assert_eq!(meta.get(&Edn::Key("bar")), Some(&Edn::Bool(false)));
        }
        _ => panic!("Expected Meta variant"),
    }
}

#[test]
fn test_symbol_tag_shorthand() {
    let result = read_string("^String x").unwrap();
    match result {
        Edn::Meta(meta, value) => {
            assert_eq!(*value, Edn::Symbol("x"));
            assert_eq!(meta.get(&Edn::Key("tag")), Some(&Edn::Symbol("String")));
        }
        _ => panic!("Expected Meta variant"),
    }
}
```

### Step 7: Update Our Integration

**File**: `src/reader.rs` (our project)

Update `edn_to_value` to handle metadata:

```rust
fn edn_to_value(edn: &Edn) -> Result<Value, String> {
    match edn {
        // ... existing cases

        // NEW: Handle metadata
        Edn::Meta(meta_map, value) => {
            // Convert metadata to our HashMap format
            let mut metadata = im::HashMap::new();
            for (k, v) in meta_map {
                let key = match k {
                    Edn::Key(s) => s.to_string(),
                    Edn::Symbol(s) => s.to_string(),
                    _ => continue, // Skip non-string keys
                };
                metadata.insert(key, edn_to_value(v)?);
            }

            // Wrap the value with metadata
            // We'll need to enhance our Value enum to carry metadata
            let inner = edn_to_value(value)?;
            Ok(Value::WithMeta(metadata, Box::new(inner)))
        }

        // ... rest of cases
    }
}
```

We'd also need to update our `Value` enum:

```rust
// src/value.rs
pub enum Value {
    // ... existing variants

    // NEW: Carry metadata
    WithMeta(im::HashMap<String, Value>, Box<Value>),
}
```

## Changes to Our Codebase

### 1. Update `Cargo.toml`

Replace the crates.io dependency with our fork:

```toml
[dependencies]
clojure-reader = { git = "https://github.com/YOUR_USERNAME/clojure-reader", branch = "metadata-support" }
```

### 2. Update `src/value.rs`

Add `WithMeta` variant:

```rust
pub enum Value {
    Number(isize),
    Bool(bool),
    Keyword(String),
    Symbol(String),
    String(String),
    List(im::Vector<Value>),
    Vector(im::Vector<Value>),
    Map(im::HashMap<Value, Value>),
    Nil,

    // NEW
    WithMeta(im::HashMap<String, Value>, Box<Value>),
}
```

### 3. Update `src/clojure_ast.rs`

Remove the earmuff convention hack in `analyze_def`:

```rust
fn analyze_def(items: &im::Vector<Value>) -> Result<Expr, String> {
    if items.len() != 3 {
        return Err(format!("def requires 2 arguments, got {}", items.len() - 1));
    }

    // Extract metadata if present
    let (name, metadata) = match &items[1] {
        Value::WithMeta(meta, inner) => {
            match **inner {
                Value::Symbol(ref s) => (s.clone(), Some(meta.clone())),
                _ => return Err("def requires a symbol".to_string()),
            }
        }
        Value::Symbol(s) => (s.clone(), None),
        _ => return Err("def requires a symbol as first argument".to_string()),
    };

    let value = analyze(&items[2])?;

    Ok(Expr::Def {
        name,
        value: Box::new(value),
        metadata,
    })
}
```

## Effort Breakdown

### Forking and Setup (1 day)
- Fork repository on GitHub
- Set up local development environment
- Run existing tests to ensure they pass
- Create `metadata-support` branch

### Core Implementation (2 days)
- Add `Edn::Meta` variant
- Implement `parse_metadata` function
- Handle all shorthand forms
- Implement metadata merging for chaining
- Update Display implementation

### Testing (1 day)
- Write comprehensive tests for all metadata forms
- Test chaining behavior
- Test error cases
- Ensure existing tests still pass

### Integration (1 day)
- Update our `reader.rs` to handle metadata
- Add `WithMeta` to our `Value` enum
- Update `analyze_def` to extract metadata
- Test end-to-end with our project
- Update documentation

## Risks and Mitigations

### Risk 1: Breaking Changes
**Mitigation**: Add `Meta` variant behind a feature flag initially:
```rust
#[cfg(feature = "metadata")]
Meta(BTreeMap<Edn<'e>, Edn<'e>>, Box<Edn<'e>>),
```

### Risk 2: Maintenance Burden
**Mitigation**:
- Submit PR upstream to clojure-reader
- If accepted, switch back to crates.io version
- If rejected, maintain minimal fork

### Risk 3: Performance Impact
**Mitigation**:
- Metadata is only allocated when `^` is encountered
- No impact on code without metadata
- Box keeps size of Edn enum reasonable

## Alternative: EDN Spec Compliance

**Important Note**: The [EDN specification](https://github.com/edn-format/edn) explicitly states:

> "The metadata syntax is not a feature of EDN. It is valid only in Clojure."

This means:
- EDN parsers are **not required** to support `^`
- Metadata is a **Clojure extension** beyond EDN
- The crate is called `clojure-reader`, so metadata support is appropriate

## Recommendation

**YES, fork and add metadata support.** Here's why:

### Pros
✅ True Clojure syntax compatibility
✅ Better developer experience
✅ Removes earmuff convention hack
✅ Enables other metadata features (`:private`, `:doc`, etc.)
✅ Moderate complexity, well-scoped work
✅ Could be upstreamed to benefit community

### Cons
❌ Maintenance burden if not upstreamed
❌ 3-5 days of work
❌ Need to track upstream changes

### Decision Criteria

**Fork if**:
- You plan to add more Clojure features (beyond POC)
- You want 100% syntax compatibility
- You're willing to maintain a fork or submit PR

**Keep earmuff convention if**:
- This is strictly a POC (proof of concept)
- Time is limited
- Current solution works perfectly well

## Next Steps

1. **Create fork**: Fork https://github.com/Grinkers/clojure-reader
2. **Create branch**: `git checkout -b metadata-support`
3. **Implement**: Follow the implementation plan above
4. **Test**: Write comprehensive tests
5. **Integrate**: Update our project to use the fork
6. **PR**: Consider submitting upstream PR

## Sources

- [Clojure Reader Documentation](https://clojure.org/reference/reader)
- [Clojure Metadata Documentation](https://clojure.org/reference/metadata)
- [clojure-reader GitHub](https://github.com/Grinkers/clojure-reader)
- [EDN Specification](https://github.com/edn-format/edn)
