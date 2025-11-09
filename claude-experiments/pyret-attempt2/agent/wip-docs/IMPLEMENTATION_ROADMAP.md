# Implementation Roadmap - Quick Reference

**Purpose:** Fast reference for what to implement next
**Full Details:** See `COMPREHENSIVE_GAP_ANALYSIS.md`
**Test Suite:** `tests/comprehensive_gap_tests.rs` (50+ tests)

---

## Current Status

✅ **Basic Parser:** 76/81 tests passing (93.8%)
⚠️ **Advanced Features:** 50+ tests waiting (all marked `#[ignore]`)

**Combined Completion:** ~60% of production-ready parser

---

## Quick Start - What to Build Next

### 🔥 Critical Path (Do This First)

#### 1. Data Definitions (6-8 hours) ⭐⭐⭐⭐⭐
```pyret
data Either:
  | left(v)
  | right(v)
end
```
**Why:** Core Pyret feature, required for cases expressions
**Tests:** 6 tests in `test_data_*`

#### 2. Cases Expressions (4-6 hours) ⭐⭐⭐⭐⭐
```pyret
cases (Either) e:
  | left(v) => v
  | right(v) => v
end
```
**Why:** Pattern matching, works with data definitions
**Tests:** 4 tests in `test_cases_*`

#### 3. Advanced Blocks (3-4 hours) ⭐⭐⭐⭐
```pyret
block:
  x = 10
  y = 20
  x + y
end
```
**Why:** Multiple statements in blocks, common pattern
**Tests:** 4 tests in `test_block_*`

**Total:** 13-18 hours, unlocks 14 tests, gets to ~70% complete

---

## Feature Checklist

Use this checklist to track implementation progress:

### Core Language
- [ ] Data definitions (6 tests)
  - [ ] Simple enumerations
  - [ ] Variants with fields
  - [ ] Mutable fields (`ref`)
  - [ ] Multiple variants
  - [ ] Sharing clauses
  - [ ] Generic types
- [ ] Cases expressions (4 tests)
  - [ ] Basic pattern matching
  - [ ] Else branches
  - [ ] Nested cases
  - [ ] Cases in functions
- [ ] Advanced blocks (4 tests)
  - [ ] Multiple let bindings
  - [ ] Var bindings
  - [ ] Type annotations on bindings
  - [ ] Nested blocks with shadowing

### Testing & Functions
- [ ] Check blocks (2 tests)
  - [ ] Standalone checks
  - [ ] Example-based testing
- [ ] Advanced functions (4 tests)
  - [ ] Multiple where clauses
  - [ ] Recursive with cases
  - [ ] Higher-order functions
  - [ ] Rest parameters
- [ ] Advanced for expressions (4 tests)
  - [ ] Multiple generators
  - [ ] Fold with complex accumulators
  - [ ] For filter
  - [ ] Nested for

### Type System
- [ ] Type annotations (3 tests)
  - [ ] Arrow types
  - [ ] Union types
  - [ ] Generic type parameters
- [ ] Contracts (1 test)

### Special Features
- [ ] Table expressions (2 tests)
- [ ] String interpolation (2 tests)
- [ ] Operators (3 tests)
  - [ ] Custom operators
  - [ ] Unary not
  - [ ] Unary minus
- [ ] Spy expressions (1 test)

### Module System
- [ ] Import/export (4 tests)
  - [ ] Import with aliases
  - [ ] Import from files
  - [ ] Provide-types
  - [ ] Selective exports
- [ ] Advanced objects (3 tests)
  - [ ] Object extension
  - [ ] Computed properties
  - [ ] Object updates

### Advanced
- [ ] Comprehensions (1 test)
- [ ] Real-world integration (2 tests)
- [ ] Gradual typing (1 test)

---

## Implementation Patterns

### Adding a New Expression Type

1. **Add AST node** in `src/ast.rs`
2. **Add parser method** in `src/parser.rs`
3. **Update expression parser** to call new method
4. **Add JSON serialization** in `src/bin/to_pyret_json.rs`
5. **Add location extraction** in helper functions
6. **Write unit test** in `tests/parser_tests.rs`
7. **Enable comparison test** - remove `#[ignore]`
8. **Verify with Pyret** - run `./compare_parsers.sh`

### Example: Implementing Data Definitions

```rust
// 1. AST node (src/ast.rs)
pub struct SData {
    pub l: Loc,
    pub name: String,
    pub params: Vec<String>,
    pub variants: Vec<Variant>,
    pub shared: Vec<Member>,
}

// 2. Parser method (src/parser.rs)
fn parse_data_expr(&mut self) -> Result<Expr> {
    self.expect(TokenType::Data)?;
    let name = self.expect_name()?;
    self.expect(TokenType::Colon)?;
    // ... parse variants
    self.expect(TokenType::End)?;
    Ok(Expr::SData { /* ... */ })
}

// 3. Update expression parser
fn parse_prim_expr(&mut self) -> Result<Expr> {
    match self.current_token() {
        TokenType::Data => self.parse_data_expr(),
        // ... other cases
    }
}

// 4. JSON serialization (src/bin/to_pyret_json.rs)
Expr::SData { name, variants, .. } => json!({
    "type": "s-data",
    "name": name,
    "variants": variants.iter().map(variant_to_json).collect::<Vec<_>>(),
    // ...
})
```

---

## Testing Workflow

### Before Implementation
```bash
# See all ignored tests
cargo test --test comprehensive_gap_tests -- --ignored --list

# Run specific category
cargo test --test comprehensive_gap_tests test_data -- --ignored
```

### During Implementation
```bash
# Remove #[ignore] from test
# Run single test
cargo test --test comprehensive_gap_tests test_simple_data_definition

# Compare with Pyret
./compare_parsers.sh "data Either: | left(v) | right(v) end"
```

### After Implementation
```bash
# Run all tests in category
cargo test --test comprehensive_gap_tests test_data

# Run full test suite
cargo test
```

---

## Time Estimates by Phase

| Phase | Features | Tests | Time | Completion |
|-------|----------|-------|------|------------|
| **Current** | Basic parser | 76 | - | 60% |
| **Phase 1** | Data, Cases, Blocks | 14 | 13-18h | 70% |
| **Phase 2** | Checks, Functions, For | 10 | 7-10h | 75% |
| **Phase 3** | Types, Contracts | 4 | 6-8h | 78% |
| **Phase 4** | Tables, Strings, Ops | 8 | 9-13h | 82% |
| **Phase 5** | Modules, Objects | 7 | 7-10h | 86% |
| **Phase 6** | Advanced patterns | 4 | 2-3h | 90% |
| **Complete** | All features | 50+ | 44-62h | 95% |

---

## Success Metrics

### Test Coverage
- **Now:** 76/81 basic tests (93.8%)
- **Phase 1:** 90/131 tests (~69%)
- **Complete:** 126/131 tests (~96%)

### Feature Coverage
- **Now:** Core expressions ✅
- **Phase 1:** Algebraic data types ✅
- **Complete:** Production-ready parser ✅

---

## Common Issues & Solutions

### Issue: AST doesn't match Pyret
**Solution:** Run `./compare_parsers.sh "code"` and examine differences
- Check field names in JSON output
- Verify field order matches
- Ensure location info is present

### Issue: Parser panics
**Solution:** Add better error handling
- Check token types before consuming
- Add helpful error messages
- Use `expect()` for required tokens

### Issue: Test still fails after implementation
**Solution:** Debug step by step
1. Check if tokenizer recognizes keywords
2. Verify parser method is called
3. Print parsed AST
4. Compare with Pyret JSON output
5. Adjust field names/structure

---

## Resources

### Documentation
- `COMPREHENSIVE_GAP_ANALYSIS.md` - Full feature analysis
- `NEXT_STEPS.md` - Detailed implementation guides
- `PARSER_GAPS.md` - Original gap analysis

### Code References
- `src/parser.rs` - Parser implementation
- `src/ast.rs` - AST definitions
- `src/bin/to_pyret_json.rs` - JSON serialization
- `tests/parser_tests.rs` - Unit tests
- `tests/comprehensive_gap_tests.rs` - Integration tests

### External References
- Pyret Grammar: `/path/to/pyret-lang/src/js/base/pyret-grammar.bnf`
- Pyret Examples: `/path/to/pyret-lang/tests/pyret/tests/*.arr`
- Comparison Script: `./compare_parsers.sh`

---

## Quick Commands

```bash
# See what needs to be done
cargo test --test comprehensive_gap_tests -- --ignored --list

# Work on data definitions
cargo test --test comprehensive_gap_tests test_data -- --ignored

# Implement feature, remove #[ignore], then:
cargo test --test comprehensive_gap_tests test_simple_data_definition

# Validate against Pyret
./compare_parsers.sh "data Box: | box(v) end"

# Run all tests
cargo test

# See full progress
cargo test --test comprehensive_gap_tests -- --ignored | grep "test result"
```

---

## Questions?

1. **What should I implement first?**
   → Data definitions (Phase 1, most impactful)

2. **How do I know if it's working?**
   → Test passes AND `./compare_parsers.sh` shows identical AST

3. **What if I get stuck?**
   → Look at similar features (e.g., `SLam` for `SData`)

4. **How much time will this take?**
   → Phase 1: 13-18 hours, Full: 44-62 hours

5. **Where do I start?**
   → Read `test_simple_data_definition` in `comprehensive_gap_tests.rs`

---

**Last Updated:** $(date)
**Next Review:** After Phase 1 completion
**Maintained By:** Development Team
