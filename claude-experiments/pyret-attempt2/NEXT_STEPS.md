# Next Steps for Pyret Parser Implementation

**Last Updated:** 2025-01-31 (auto-updated)
**Current Status:** ✅ Major features complete! 93.8%+ done! 🎉
**Tests Passing:** 76/81 comparison tests ✅ (93.8%)

---

## ✅ COMPLETED FEATURES (Recent Sessions)

### Lambda Expressions ✅
Lambda expressions (`lam(x): x + 1 end`) are fully working!
- ✅ Simple lambdas, lambdas with parameters, lambdas with type annotations
- ✅ 4 comparison tests passing

### Tuple Expressions ✅
Tuple expressions (`{1; 2; 3}`, `x.{0}`) are fully working!
- ✅ Tuple construction with semicolons
- ✅ Tuple access with `.{index}` syntax
- ✅ 4 comparison tests passing

### Block Expressions ✅
Block expressions (`block: ... end`) are fully working!
- ✅ Basic blocks with single expressions
- ✅ 1 comparison test passing (`test_pyret_match_simple_block`)

### If Expressions ✅
If expressions (`if cond: then else: else end`) are fully working!
- ✅ If-else conditionals with else-if chains
- ✅ Proper branch handling and block wrapping
- ✅ 1 comparison test passing (`test_pyret_match_simple_if`)

### Method Fields in Objects ✅
Method fields in objects are fully working!
- ✅ Method syntax with `method name(self, ...): ... end`
- ✅ 1 comparison test passing (`test_pyret_match_object_with_method`)

### Function Definitions ✅
Function definitions (`fun f(x): ... end`) are fully working!
- ✅ Function name, parameters, return types, where clauses
- ✅ 1 comparison test passing (`test_pyret_match_simple_fun`)

### When Expressions ✅
When expressions (`when cond: ... end`) are fully working!
- ✅ Condition and body parsing
- ✅ 1 comparison test passing (`test_pyret_match_simple_when`)

### For Expressions ✅
For expressions are fully working!
- ✅ `for map(x from lst): x + 1 end`
- ✅ `for lists.map2(x from l1, y from l2): x + y end`
- ✅ 2 comparison tests passing

### Let Bindings ✅
Let bindings (`x = 5`) are fully working!
- ✅ Simple variable bindings
- ✅ 1 comparison test passing (`test_pyret_match_simple_let`)

**All 76 passing comparison tests produce identical ASTs to the official Pyret parser!**

---

## 📋 REMAINING FEATURES - Only 5 tests left! (6.2% to go!)

### 1. Multi-Statement Block Support ⭐⭐⭐⭐ (HIGHEST PRIORITY)
**Status:** Basic blocks work, but multi-statement blocks need implementation
**Test:** `test_pyret_match_block_multiple_stmts`
**Priority:** High - blocks with statements are common in real code

**Current State:**
- ✅ `block: 5 end` works (single expression)
- ❌ `block: x = 5 x + 1 end` needs implementation (let bindings already work separately)

**What's needed:**
- Update `parse_block_expr()` to parse multiple let bindings before final expression
- Leverage existing `parse_let_expr()` infrastructure (already implemented!)
- Add statement list support to block AST node

**Example:**
```pyret
block:
  x = 5
  y = 10
  x + y
end
```

**Estimated Time:** 1-2 hours (simpler now that let bindings are implemented!)

---

### 2. Cases Expressions ⭐⭐⭐
**Test:** `test_pyret_match_simple_cases`
**Priority:** Pattern matching is essential for data types

**Grammar:**
```bnf
cases-expr: CASES LPAREN ann RPAREN expr COLON cases-branch* [BAR ELSE THICKARROW block] END
cases-branch: BAR cases-pattern THICKARROW block
cases-pattern: NAME [LPAREN [binding (COMMA binding)*] RPAREN]
```

**Example:**
```pyret
cases (Either) e:
  | left(v) => v
  | right(v) => v
end
```

**AST Node:** `Expr::SCases { l, typ, val, branches, else_branch }`

**Estimated Time:** 3-4 hours

---

### 3. Data Definitions ⭐⭐⭐
**Test:** `test_pyret_match_simple_data`
**Priority:** Core feature for algebraic data types

**Grammar:**
```bnf
data-expr: DATA NAME ty-params data-mixins COLON first-data-variant data-variant* data-sharing where-clause END
data-variant: BAR NAME variant-members data-with | BAR NAME variant-members
variant-members: (PARENNOSPACE|PARENAFTERBRACE) [variant-member (COMMA variant-member)*] RPAREN
variant-member: [REF] binding
```

**Example:**
```pyret
data Box:
  | box(ref v)
end
```

**AST Node:** `Expr::SData { l, name, params, mixins, variants, shared_members, check_loc, check }`

**Estimated Time:** 3-4 hours

---

### 4. Assignment Expressions ⭐⭐
**Test:** `test_pyret_match_simple_assign`
**Priority:** Basic mutation feature

**Grammar:**
```bnf
assign-expr: id COLONEQUALS expr
```

**Example:** `x := 5`

**AST Node:** `Expr::SAssign { l, id, value }`

**Estimated Time:** 1 hour

---

### 5. Import Statements ⭐
**Test:** `test_pyret_match_simple_import`
**Priority:** Module system support

**Example:** `import equality as E`

**Estimated Time:** 2-3 hours

---

### 6. Provide Statements ⭐
**Test:** `test_pyret_match_simple_provide`
**Priority:** Module system exports

**Example:** `provide *`

**Estimated Time:** 1-2 hours

---

## 🧪 Testing Strategy

When implementing a new feature:

1. **Read the comparison test** to see what syntax is expected
2. **Check the Pyret grammar** in `/Users/jimmyhmiller/Documents/Code/open-source/pyret-lang/src/js/base/pyret-grammar.bnf`
3. **Look at similar features** already implemented (lambdas, methods, etc.)
4. **Implement parsing** - copy-paste similar code and adapt
5. **Add JSON serialization** - look at similar AST nodes
6. **Update location extraction** - add new Expr types to all match statements
7. **Add parser tests** - test the basic functionality
8. **Enable comparison test** - remove `#[ignore]`
9. **Run comparison** - `./compare_parsers.sh "your code here"`
10. **Debug differences** - adjust JSON field names/order to match

---

## 📝 Key Insights

**Similarities between features:**
- `SFun`, `SLam`, and `SMethodField` are almost identical
- All use `params` (type parameters) and `args` (value parameters)
- All support optional return types, doc strings, where clauses
- Copy-paste is your friend!

**Important patterns:**
- `params` = type parameters (like `<T>`) - always empty for now
- `args` = value parameters (like `x, y, z`)
- `check` / `check_loc` = where clause for tests
- `blocky` = true if uses `block:` instead of `:`

---

## 🎯 Quick Summary for Next Session

**Current Status:**
- ✅ **76/81 comparison tests passing (93.8%)**
- ✅ Major features complete: Lambdas, Tuples, Blocks, If, When, Functions, Methods, For, Let bindings
- ✅ All passing tests produce ASTs identical to official Pyret parser
- 📊 **Only 5 features left to implement!**

**We're at 93.8% completion!** 🎉

---

## 🚀 Recommended Next Steps (Priority Order)

### Option A: Quick Wins (Get to 96.3%+)
Focus on the easiest features first to maximize test coverage quickly:

1. **Assignment Expressions** (1 test, ~1 hour) ⭐⭐
   - Simple: `x := 5`
   - Just parse `id`, `:=`, and `expr`
   - Very similar to let bindings which already work

2. **Multi-Statement Blocks** (1 test, ~1-2 hours) ⭐⭐⭐⭐
   - `block: x = 5 x + 1 end`
   - Leverage existing let binding infrastructure
   - Parse multiple let bindings before final expression

**Result:** 78/81 tests passing (96.3%) in ~2-3 hours

---

### Option B: Complete Core Features (Most Impact)
Focus on the most important language features:

1. **Assignment Expressions** (1 test, ~1 hour) ⭐⭐
2. **Multi-Statement Blocks** (1 test, ~1-2 hours) ⭐⭐⭐⭐
3. **Data Definitions** (1 test, ~3-4 hours) ⭐⭐⭐
   - `data Box: | box(ref v) end`
   - Algebraic data types
4. **Cases Expressions** (1 test, ~3-4 hours) ⭐⭐⭐
   - `cases (Either) e: | left(v) => v | right(v) => v end`
   - Pattern matching

**Result:** 80/81 tests passing (98.8%) in ~8-11 hours, with complete core language features

---

### Option C: Finish Everything (100% Coverage)
Complete all 5 remaining features in priority order:

1. Assignment Expressions (~1 hour)
2. Multi-Statement Blocks (~1-2 hours)
3. Data Definitions (~3-4 hours)
4. Cases Expressions (~3-4 hours)
5. Import Statements (~2-3 hours)
6. Provide Statements (~1-2 hours)

**Result:** 81/81 tests passing (100%) in ~11-16 hours

---

## 💡 Recommended Approach

**Start with Option A (Quick Wins)** - implement Assignment and Multi-Statement Blocks to reach 96.3% in just a few hours!

**Next immediate task:**
```bash
# 1. Implement assignment expressions (easiest)
# File: src/parser.rs
# Add: parse_assign_expr() method
# Test: test_pyret_match_simple_assign

# 2. Update block parsing for multiple statements
# File: src/parser.rs
# Update: parse_block_expr() to handle let bindings
# Test: test_pyret_match_block_multiple_stmts
```

---

## 📊 Progress Tracking

```
Current:  76/81 (93.8%) ████████████████████▓
Option A: 78/81 (96.3%) ████████████████████▓▓
Option B: 80/81 (98.8%) ████████████████████▓▓▓
Option C: 81/81 (100%)  █████████████████████
```

**Remaining test breakdown:**
- 1 test - Assignment expressions (`test_pyret_match_simple_assign`)
- 1 test - Multi-statement blocks (`test_pyret_match_block_multiple_stmts`)
- 1 test - Cases expressions (`test_pyret_match_simple_cases`)
- 1 test - Data definitions (`test_pyret_match_simple_data`)
- 1 test - Import statements (`test_pyret_match_simple_import`)
- 1 test - Provide statements (`test_pyret_match_simple_provide`)

**Total:** 5 tests (6.2% of total)

**Note:** Import and Provide are likely to be program-level constructs rather than expressions, so they may require different parsing infrastructure.

