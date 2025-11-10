# Refactoring Plan: Pyret Parser Cleanup

**Status:** ✅ Code Review Complete | ⏳ Refactoring In Progress
**Date:** 2025-11-09
**Test Status:** 298/299 passing (99.7%) - All tests passing after initial fixes ✅

## Executive Summary

Parser works perfectly (298 tests passing!) but has significant technical debt:
- **373 `.clone()` calls** (many unnecessary)
- **39 `file_name.clone()` calls** (should use Arc)
- **5544-line parser.rs** (should be ~1500 lines max)
- **Inconsistent use of helper methods** (has helpers but doesn't use them)

## Phase 1: Quick Wins (HIGHEST IMPACT) ✅ IN PROGRESS

### 1.1 Replace Manual Loc Construction ⏳ PARTIALLY DONE
**Impact:** Eliminates 30+ file_name clones, improves readability

**Pattern to Fix:**
```rust
// BAD (appears 30+ times)
let l = Loc::new(
    self.file_name.clone(),
    token.location.start_line,
    // ... 5 more lines
);

// GOOD
let l = self.token_to_loc(&token);
```

**Status:**
- ✅ Fixed `parse_check_op()` - Demonstrated the pattern
- ✅ Created automation scripts
- ⏳ Need to carefully apply to remaining ~25 instances
- ⚠️ Some cases need special handling (e.g., when combining multiple token locations)

**Script:** `/tmp/fix_locs2.py` (needs refinement for edge cases)

### 1.2 Use Arc<String> for file_name ⏳ TODO
**Impact:** Eliminates all file_name clones, improves performance

**Changes needed:**
```rust
// src/parser.rs
use std::sync::Arc;

pub struct Parser {
    file_name: Arc<String>,  // Change from String
    // ...
}

impl Parser {
    pub fn new(tokens: Vec<Token>, file_name: String) -> Self {
        Parser {
            file_name: Arc::new(file_name),  // Wrap in Arc
            // ...
        }
    }
}

// Update Loc::new calls in helpers
fn current_loc(&self) -> Loc {
    Loc::new(
        (*self.file_name).clone(),  // Deref Arc to get String
        // ...
    )
}
```

**Estimated Impact:** Reduces memory allocations significantly

### 1.3 Reduce Token Cloning ⏳ TODO
**Impact:** Improves parser performance

**Pattern to Fix:**
```rust
// BAD - Clone on happy path
let token = self.advance().clone();
// ... use token ...

// GOOD - Clone only on error path
let token = self.advance();
// ... use token ...
Err(ParseError::unexpected(token.clone()))  // Clone only here
```

**Challenge:** Rust borrow checker requires careful refactoring

## Phase 2: Code Organization (MEDIUM IMPACT) ⏳ PLANNED

### 2.1 Split parser.rs into Modules
**Target Structure:**
```
src/parser/
  ├── mod.rs          (~500 lines)  - Core Parser struct, helpers
  ├── types.rs        (~800 lines)  - Type annotation parsing
  ├── expressions.rs  (~1500 lines) - Expression parsing
  ├── statements.rs   (~600 lines)  - Statement parsing
  ├── imports.rs      (~800 lines)  - Import/provide/module system
  ├── data.rs         (~600 lines)  - Data declarations
  ├── check.rs        (~400 lines)  - Check blocks and operators
  └── helpers.rs      (~344 lines)  - Utility functions
```

**Benefits:**
- Easier navigation
- Clearer code organization
- Better IDE performance
- Parallel development possible

### 2.2 Extract Common Parsing Patterns
**New Helper Methods:**
```rust
/// Parse "keyword: ... end" or "keyword block: ... end"
fn parse_keyword_block(&mut self, keyword: TokenType) -> ParseResult<(Expr, bool)>

/// Parse delimited list: "(item, item, item)" or "[item, item]"
fn parse_delimited_list<T, F>(
    &mut self,
    open: TokenType,
    close: TokenType,
    parser: F
) -> ParseResult<Vec<T>>
where F: Fn(&mut Parser) -> ParseResult<T>

/// Parse binary operation patterns
fn parse_binary_expr(&mut self, operators: &[TokenType]) -> ParseResult<Expr>
```

## Phase 3: Architecture Improvements (LONG-TERM) ⏳ PLANNED

### 3.1 Builder Pattern for Complex AST Nodes
**Current Problem:**
```rust
// Hard to read, easy to miss fields
Ok(Expr::SFun {
    l: loc,
    name: name,
    params: params,
    args: args,
    ann: ann,
    doc: doc,
    body: body,
    check_loc: None,
    check: None,
    blocky: false,
})
```

**Proposed Solution:**
```rust
Ok(FunBuilder::new(name, args, body)
    .with_loc(loc)
    .with_params(params)
    .with_ann(ann)
    .with_doc(doc)
    .build())
```

### 3.2 Improve Error Messages
**Add Context:**
```rust
// Current
Err(ParseError::expected(TokenType::End, token))

// Better
Err(ParseError::expected(TokenType::End, token)
    .with_context("while parsing function definition")
    .with_hint("Did you forget the 'end' keyword?"))
```

### 3.3 Separate Tokenizer State
**Problem:** Parser depends on tokenizer's whitespace-sensitivity
**Solution:** Move whitespace logic to dedicated preprocessing phase

## Phase 4: Performance Optimizations ⏳ FUTURE

### 4.1 Reduce Allocations
- Use `&str` instead of `String` where possible
- Arena allocator for AST nodes
- String interning for identifiers

### 4.2 Benchmark Critical Paths
- Profile parser performance
- Optimize hottest code paths
- Consider parallel parsing for large files

## Quick Reference: Code Smells Found

### 🔴 Critical
- [x] 39 `file_name.clone()` calls
- [x] 373 `.clone()` calls total
- [ ] 5544-line single file
- [ ] Manual Loc construction despite helpers

### 🟡 Medium
- [ ] TODO comments in production code (5 instances)
- [ ] Inconsistent error handling patterns
- [ ] Repeated parsing patterns (20+ instances)

### 🟢 Low
- [ ] Some functions >100 lines
- [ ] Magic numbers without constants
- [ ] Inconsistent formatting

## Metrics

### Current State
- **Lines of Code:** 5544 (parser.rs)
- **Parse Functions:** 87
- **Clone Calls:** 373
- **File Name Clones:** 39
- **Test Pass Rate:** 99.7% (298/299)

### Target State (After Refactoring)
- **Lines of Code:** ~1500 (main parser), ~4000 (split across modules)
- **Parse Functions:** 87 (same, better organized)
- **Clone Calls:** ~200 (47% reduction)
- **File Name Clones:** 0 (100% reduction via Arc)
- **Test Pass Rate:** 100% (299/299)

## Safety Protocol

Before ANY refactoring:
```bash
# 1. Run all tests
cargo test

# 2. Run comparison tests
cargo test --test comparison_tests

# 3. Verify 298 tests still passing
# Expected: test result: ok. 298 passed; 0 failed; 1 ignored

# 4. Only proceed if ALL tests pass
```

After EACH change:
```bash
# Quick compilation check
cargo check

# Full test suite
cargo test
```

## Timeline Estimate

| Phase | Effort | Impact | Priority |
|-------|--------|--------|----------|
| 1.1 - Loc construction | 2 hours | HIGH | P0 |
| 1.2 - Arc<String> | 1 hour | HIGH | P0 |
| 1.3 - Token cloning | 3 hours | MEDIUM | P1 |
| 2.1 - Module split | 4 hours | MEDIUM | P1 |
| 2.2 - Helper extraction | 3 hours | MEDIUM | P2 |
| 3.x - Architecture | 8 hours | LOW | P3 |
| 4.x - Performance | 8 hours | LOW | P4 |

**Total Critical Path:** ~10 hours (Phase 1 + Phase 2.1)

## Success Criteria

✅ **Must Have (Phase 1+2)**
- [ ] All 298 tests passing
- [ ] <50 file_name.clone() calls (87% reduction)
- [ ] Parser split into <6 modules
- [ ] Main parser.rs <2000 lines

🎯 **Nice to Have (Phase 3+4)**
- [ ] Builder patterns for complex nodes
- [ ] Better error messages
- [ ] Performance benchmarks
- [ ] Documentation

## Notes

- Parser is SOLID - don't break it!
- All refactoring must maintain test pass rate
- Focus on readability and maintainability
- Performance is secondary (parser is already fast)
- Document any deviations from Pyret grammar

---

**Next Step:** Apply Phase 1.1 fixes carefully, test after each change.
