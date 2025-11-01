# 🚀 START HERE - Pyret Parser Implementation Guide

**Welcome!** This document is your entry point to implementing the missing features in the Pyret parser.

---

## 📊 Current Status

**Last Updated:** 2025-10-31

```
✅ 59 tests passing (73% coverage)
⏸️ 22 tests ignored (features to implement)
❌ 0 tests failing (no regressions!)

Time to 100%: ~30-40 hours total
Time to 86%: ~10-12 hours (Phase 1)
```

---

## 📚 Documentation Roadmap

Read these files **in order** for maximum effectiveness:

### 1️⃣ Quick Overview (5 minutes)
**👉 Start here:** `GAP_ANALYSIS_SUMMARY.md`
- What's missing (high-level)
- Why it matters
- Time estimates
- Quick commands

### 2️⃣ Detailed Analysis (15 minutes)
**👉 Then read:** `PARSER_GAPS.md`
- All 22 missing features
- Priority rankings (⭐⭐⭐⭐⭐ to ⭐)
- Official Pyret AST examples
- 3-phase implementation roadmap
- Real code references

### 3️⃣ Real Examples (10 minutes)
**👉 Browse:** `MISSING_FEATURES_EXAMPLES.md`
- Concrete Pyret code that fails
- Current vs expected behavior
- Why each feature is important
- Examples from official test files

### 4️⃣ Implementation Guide (Reference)
**👉 Keep open:** `IMPLEMENTATION_GUIDE.md`
- Step-by-step implementation process
- Feature-specific guides
- Troubleshooting section
- Test verification checklist

### 5️⃣ Original Documentation (Background)
**👉 For context:**
- `CLAUDE.md` - Project overview and status
- `NEXT_STEPS.md` - Original implementation patterns
- `README.md` - Quick reference

---

## 🎯 Your Mission

Implement **22 missing features** to get the parser from **73% → 100%** test coverage.

**Start with:** Lambda expressions (highest priority! ⭐⭐⭐⭐⭐)

---

## 🔥 Quick Start Guide

### Step 1: Understand What's Missing (5 min)

```bash
# See high-level overview
cat GAP_ANALYSIS_SUMMARY.md

# See all ignored tests
cargo test --test comparison_tests -- --ignored --list

# Output:
# test_pyret_match_simple_lambda: IGNORED
# test_pyret_match_lambda_with_params: IGNORED
# ... 22 total
```

### Step 2: Pick a Feature (1 min)

**Recommended order:**
1. Lambda expressions ⭐⭐⭐⭐⭐ (4 tests, 2-3 hours)
2. Tuple expressions ⭐⭐⭐⭐ (4 tests, 2-3 hours)
3. Block expressions ⭐⭐⭐⭐ (2 tests, 2-3 hours)
4. If expressions ⭐⭐⭐⭐ (1 test, 2-3 hours)

**Start with lambdas** - they unlock 90% of real Pyret code!

### Step 3: Learn the Feature (10 min)

```bash
# Read detailed spec
grep -A 50 "Lambda Expressions" PARSER_GAPS.md

# See real examples that fail
grep -A 30 "Lambda Expressions" MISSING_FEATURES_EXAMPLES.md

# See expected AST structure
./compare_parsers.sh "lam(): 5 end" 2>&1 | grep -A 50 "Pyret AST"
```

### Step 4: Implement (2-3 hours)

**Follow the guide in `IMPLEMENTATION_GUIDE.md`:**

1. Add parser method in `src/parser.rs`
2. Update `parse_prim_expr()` or `parse_binop_expr()`
3. Update location extraction (2 places!)
4. Update JSON serialization in `src/bin/to_pyret_json.rs`
5. Remove `#[ignore]` from tests
6. Run tests: `cargo test --test comparison_tests`

**Detailed steps:** See `IMPLEMENTATION_GUIDE.md` lines 195-350

### Step 5: Verify (5 min)

```bash
# Run your tests
cargo test --test comparison_tests test_pyret_match_simple_lambda

# Verify AST matches
./compare_parsers.sh "lam(x): x + 1 end"
# Should output: "Identical!"

# Check no regressions
cargo test --test comparison_tests
# All 59 existing tests should still pass
```

### Step 6: Repeat!

Continue with next feature. After Phase 1 (4 features, ~10-12 hours):
- **70/81 tests passing (86% coverage)**
- **Can parse real Pyret programs!** 🎉

---

## 📁 File Navigation

### Documentation (Read These)

```
START_HERE.md                      ← You are here!
GAP_ANALYSIS_SUMMARY.md            ← Read first (overview)
PARSER_GAPS.md                     ← Read second (detailed specs)
MISSING_FEATURES_EXAMPLES.md       ← Read third (real examples)
IMPLEMENTATION_GUIDE.md            ← Keep open (step-by-step guide)

CLAUDE.md                          ← Project overview
NEXT_STEPS.md                      ← Original implementation patterns
README.md                          ← Quick reference
HANDOFF_CHECKLIST.md               ← Verification checklist
```

### Source Code (Edit These)

```
src/parser.rs                      ← Add parser methods here (Section 6)
src/parser.rs:301-322              ← Update location extraction here!
src/bin/to_pyret_json.rs           ← Update JSON serialization here
src/ast.rs:292-808                 ← Reference: AST definitions

tests/comparison_tests.rs:498-706  ← Your target: 22 ignored tests
tests/parser_tests.rs              ← Optional: add unit tests here
```

### Tools & Scripts

```
./compare_parsers.sh "code"        ← Compare ASTs with official parser
cargo test --test comparison_tests ← Run comparison tests
cargo test --test parser_tests     ← Run unit tests
DEBUG_TOKENS=1 cargo test name     ← Debug tokenization
```

### Reference Materials

```
/Users/jimmyhmiller/Documents/Code/open-source/pyret-lang/
  ├── src/js/base/pyret-grammar.bnf       ← Official grammar
  └── tests/pyret/tests/
      ├── test-lists.arr                   ← Lambda examples
      ├── test-tuple.arr                   ← Tuple examples
      ├── test-binops.arr                  ← Operator examples
      └── ... more real Pyret code
```

---

## 🎯 Priority Matrix

### Phase 1: Core Expressions (HIGHEST PRIORITY) ⭐⭐⭐⭐⭐

**Goal:** Make real Pyret programs possible

| Feature | Tests | Time | Impact | Start After |
|---------|-------|------|--------|-------------|
| 1. Lambdas | 4 | 2-3h | 🔥🔥🔥🔥🔥 | NOW |
| 2. Tuples | 4 | 2-3h | 🔥🔥🔥🔥 | Lambdas |
| 3. Blocks | 2 | 2-3h | 🔥🔥🔥🔥 | Tuples |
| 4. If | 1 | 2-3h | 🔥🔥🔥🔥 | Blocks |

**Total:** 11 tests, ~10-12 hours → **86% coverage**

### Phase 2: Advanced Features ⭐⭐⭐

**Goal:** Functional programming + OOP

| Feature | Tests | Time | Impact | Start After |
|---------|-------|------|--------|-------------|
| 5. Methods | 1 | 2-3h | 🔥🔥🔥 | Phase 1 |
| 6. For | 2 | 3-4h | 🔥🔥🔥 | Methods |
| 7. Cases | 1 | 4-5h | 🔥🔥🔥 | For |

**Total:** 4 tests, ~9-12 hours → **91% coverage**

### Phase 3: Statements ⭐⭐

**Goal:** Complete language support

| Feature | Tests | Time | Start After |
|---------|-------|------|-------------|
| 8-14. Declarations, etc. | 7 | 11-17h | Phase 2 |

**Total:** 7 tests, ~11-17 hours → **100% coverage**

---

## 💡 Key Insights

### What's Actually Hard

1. **Tuple vs Object disambiguation** - `{1; 2}` (tuple) vs `{x: 1}` (object)
   - Must check first separator token
2. **Block parsing** - Requires statement infrastructure
3. **For-expression iterators** - Multiple variants (map, filter, fold)

### What's Easier Than Expected

1. **Simple lambdas** - Just `lam + params + colon + body + end`
2. **If expressions** - Straightforward conditional + branches
3. **Most postfix operators** - Pattern already established

### Critical Success Factors

1. **JSON must match EXACTLY** - Use `./compare_parsers.sh` constantly
2. **Update location extraction** - Easy to forget (2 places!)
3. **Test incrementally** - Don't implement 3 features before testing
4. **Follow existing patterns** - Study similar implementations first

---

## 🧪 Test Information

### Test File Structure

**Location:** `tests/comparison_tests.rs`

- **Lines 1-497:** 59 passing tests ✅
- **Lines 498-706:** 22 ignored tests ⏸️ (YOUR TARGET!)

### How Tests Work

Each test uses `assert_matches_pyret()`:

```rust
#[test]
#[ignore] // ← Remove this when implemented!
fn test_pyret_match_simple_lambda() {
    assert_matches_pyret("lam(): \"no-op\" end");
}
```

This function:
1. Parses with **official Pyret parser** → JSON
2. Parses with **our Rust parser** → JSON
3. Compares JSONs → Test passes if **identical**

### Running Tests

```bash
# See all ignored tests
cargo test --test comparison_tests -- --ignored --list

# Run specific ignored test (will fail until implemented)
cargo test --test comparison_tests test_pyret_match_simple_lambda -- --ignored

# Run all comparison tests (59 should pass)
cargo test --test comparison_tests

# Check for regressions
cargo test --test comparison_tests -- --skip ignored
```

---

## 🔧 Essential Commands

### Understanding a Feature

```bash
# Read spec
grep -A 50 "Lambda Expressions" PARSER_GAPS.md

# See real examples
grep -A 30 "Lambda Expressions" MISSING_FEATURES_EXAMPLES.md

# Check official AST
./compare_parsers.sh "lam(): 5 end" 2>&1 | grep -A 50 "Pyret AST"

# Look at official grammar
grep -A 10 "lam-expr" /Users/jimmyhmiller/Documents/Code/open-source/pyret-lang/src/js/base/pyret-grammar.bnf
```

### During Implementation

```bash
# Check tokenization
DEBUG_TOKENS=1 cargo test test_pyret_match_simple_lambda

# Verify AST
./compare_parsers.sh "lam(x): x + 1 end"

# Run specific test
cargo test --test comparison_tests test_pyret_match_simple_lambda

# Check for warnings
cargo clippy
```

### After Implementation

```bash
# Run all comparison tests
cargo test --test comparison_tests

# Check test count
cargo test --test comparison_tests 2>&1 | grep "test result"
# Should show: "60 passed; 0 failed; 21 ignored" (after first feature)

# Format code
cargo fmt
```

---

## 📈 Progress Tracking

### After Each Feature

1. **Count tests:**
   ```bash
   cargo test --test comparison_tests 2>&1 | grep "test result"
   ```

2. **Calculate coverage:**
   ```
   passing / 81 * 100 = coverage %
   ```

3. **Commit:**
   ```bash
   git commit -m "feat(parser): implement <feature>

   - Details...
   - Tests: X/81 passing (Y% coverage)"
   ```

### Milestones

- **Now:** 59/81 (73%) - Expression basics
- **Phase 1:** 70/81 (86%) - Real programs work!
- **Phase 2:** 74/81 (91%) - Full functional + OOP
- **Phase 3:** 81/81 (100%) - Complete language

---

## 🆘 Getting Help

### If Stuck

1. **Check similar features** - Look at existing implementations in `src/parser.rs:346-520`
2. **Read the guide** - `IMPLEMENTATION_GUIDE.md` has step-by-step instructions
3. **Compare ASTs** - Use `./compare_parsers.sh` to see expected structure
4. **Check troubleshooting** - `IMPLEMENTATION_GUIDE.md` lines 815-960

### Common Issues

- **Test fails:** JSON structure doesn't match → See `IMPLEMENTATION_GUIDE.md:821`
- **Unexpected token:** Tokenizer issue → Use `DEBUG_TOKENS=1`
- **Location wrong:** Forgot to update extraction → See `IMPLEMENTATION_GUIDE.md:855`
- **Compilation error:** AST field mismatch → Compare with `src/ast.rs`

### Reference Files

- **Parser patterns:** `src/parser.rs:188-520`
- **AST definitions:** `src/ast.rs:292-808`
- **JSON serialization:** `src/bin/to_pyret_json.rs`
- **Test examples:** `tests/comparison_tests.rs:1-497`

---

## ✅ Success Checklist

Before marking a feature complete:

- [ ] Parser method implemented in `src/parser.rs`
- [ ] Added to `parse_prim_expr()` or `parse_binop_expr()`
- [ ] Location extraction updated (2 places!)
- [ ] JSON serialization updated
- [ ] `#[ignore]` removed from tests
- [ ] Comparison tests pass
- [ ] No regressions (59 existing tests still pass)
- [ ] Manual verification with `./compare_parsers.sh`
- [ ] No compiler warnings
- [ ] Code formatted

---

## 🎓 Learning Path

### First Feature (Lambda - 2-3 hours)

1. Read `GAP_ANALYSIS_SUMMARY.md` (5 min)
2. Read Lambda section in `PARSER_GAPS.md` (10 min)
3. Read Lambda section in `MISSING_FEATURES_EXAMPLES.md` (5 min)
4. Check expected AST: `./compare_parsers.sh "lam(): 5 end"` (5 min)
5. Read implementation guide: `IMPLEMENTATION_GUIDE.md:305-495` (15 min)
6. Implement! (2-3 hours)
7. Test and verify (10 min)

**Total:** ~2-4 hours

### Subsequent Features (1-3 hours each)

You'll get faster as you learn the patterns!

---

## 🏁 Ready to Start?

**Your next command:**

```bash
cat GAP_ANALYSIS_SUMMARY.md
```

Then dive into lambdas:

```bash
grep -A 50 "Lambda Expressions" PARSER_GAPS.md
```

**Good luck!** 🚀

You have:
- ✅ 22 tests waiting for you
- ✅ Clear priorities (start with lambdas!)
- ✅ Comprehensive documentation
- ✅ Step-by-step guides
- ✅ Real examples
- ✅ Verification tools

**Everything you need is here.** Time to code! 💪

---

## 📞 Quick Reference

**Documentation Hub:**
- Overview: `GAP_ANALYSIS_SUMMARY.md`
- Detailed specs: `PARSER_GAPS.md`
- Real examples: `MISSING_FEATURES_EXAMPLES.md`
- Implementation: `IMPLEMENTATION_GUIDE.md`
- This file: `START_HERE.md`

**Test Location:**
- Ignored tests: `tests/comparison_tests.rs:498-706`

**Code Locations:**
- Add parsers: `src/parser.rs` Section 6
- Update locations: `src/parser.rs:301-322`
- Update JSON: `src/bin/to_pyret_json.rs`

**Key Commands:**
- Compare ASTs: `./compare_parsers.sh "code"`
- Run tests: `cargo test --test comparison_tests`
- Debug tokens: `DEBUG_TOKENS=1 cargo test name`

**Support:**
- Troubleshooting: `IMPLEMENTATION_GUIDE.md:815-960`
- Patterns: Look at existing parsers in `src/parser.rs`

---

**Last Updated:** 2025-10-31
**Next Step:** Read `GAP_ANALYSIS_SUMMARY.md`
