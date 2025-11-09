# Pyret Parser - Gap Analysis

**Generated:** 2025-11-01
**Test Results:** 69 passing, 12 missing features (ignored tests)

This document identifies missing features in the Pyret parser based on comparison tests against the official Pyret parser using **real Pyret code** from the official repository.

---

## 📊 Summary

**Implemented (69 tests passing):**
- ✅ All primitive expressions (numbers, strings, booleans, identifiers)
- ✅ Binary operators (15 operators, left-associative, NO precedence)
- ✅ Parenthesized expressions
- ✅ Function application (direct and chained)
- ✅ Whitespace-sensitive parsing (`f(x)` vs `f (x)`)
- ✅ Dot access (single and chained)
- ✅ Bracket access (`arr[0]`, `matrix[i][j]`)
- ✅ Construct expressions (`[list: 1, 2, 3]`, `[set: x, y]`)
- ✅ Check operators (`is`, `raises`, `satisfies`, `violates`)
- ✅ Object expressions (`{ x: 1, y: 2 }`)
- ✅ Lambda expressions (`lam(x): x + 1 end`)
- ✅ Tuple expressions (`{1; 2; 3}`, `x.{2}`)
- ✅ Block expressions (`block: ... end`) - NEW! ✨
- ✅ If expressions (`if cond: ... else: ... end`) - NEW! ✨
- ✅ Postfix operator chaining

**Not Yet Implemented (12 ignored tests):**
- ❌ For expressions
- ❌ Method fields in objects
- ❌ Cases expressions
- ❌ When expressions
- ❌ Assignment expressions
- ❌ Let/Var bindings
- ❌ Data declarations
- ❌ Function declarations
- ❌ Import/Provide statements

---

## 🎯 Priority 1: Core Expressions (High Impact)

These are fundamental expression types used in almost every Pyret program.

### 1. ✅ Block Expressions (2 tests) - COMPLETED!

**Status:** ✅ Fully implemented in Session 2025-11-01 PART 1

**Priority:** ⭐⭐⭐⭐ (Essential for control flow)

**Syntax:**
```pyret
block: expr end
block: stmt1 stmt2 expr end
```

**Real Examples from real Pyret code:**
```pyret
block:
  x = 5
  y = 10
  x + y
end

lam(arr) block:
  for each(x from arr):
    print(x)
  end
end
```

**AST Node:** `Expr::SUserBlock`

**Implementation Notes:**
- Token: `TokenType::Block`
- Grammar: `BLOCK COLON stmts END`
- Contains multiple statements/bindings
- Last statement is the return value

**Estimated Time:** 2-3 hours

**Test Files:**
- `test_pyret_match_simple_block` ← `block: 5 end`
- `test_pyret_match_block_multiple_stmts` ← `block: x = 5 x + 1 end`

**Official Pyret AST Example:**
```json
{
  "type": "s-user-block",
  "body": {
    "type": "s-block",
    "stmts": [{"type": "s-num", "value": "5"}]
  }
}
```

---

### 2. ✅ If Expressions (1 test) - COMPLETED!

**Status:** ✅ Fully implemented in Session 2025-11-01 PART 2

**Priority:** ⭐⭐⭐⭐ (Core control flow)

**Syntax:**
```pyret
if cond: then-expr else: else-expr end
if cond: then-expr else if cond2: expr2 else: expr3 end
```

**AST Node:** `Expr::SIfElse` (with else) or `Expr::SIf` (without else)

**Implementation Details:**
- Token: `TokenType::If`
- Grammar: `IF expr COLON body (ELSEIF expr COLON body)* (ELSE COLON body)? END`
- Creates `IfBranch` structures with test and body
- Bodies wrapped in `SBlock` for proper statement handling
- Added `parse_if_expr()` method in Section 7 (Control Flow)
- JSON serialization with `if_branch_to_pyret_json()` helper

**Test Files:**
- ✅ `test_pyret_match_simple_if` ← `if true: 1 else: 2 end`

---

## 🎯 Priority 2: Advanced Expression Features

### 3. For Expressions (2 tests) - NOW HIGHEST PRIORITY!

**Priority:** ⭐⭐⭐ (Functional programming essential)

**Syntax:**
```pyret
for map(x from lst): x + 1 end
for lists.map2(x from lst1, y from lst2): x + y end
for filter(x from lst): x > 5 end
for fold(acc from init, x from lst): acc + x end
```

**Real Example from `test-binops.arr`:**
```pyret
for lists.map2(a1 from self.arr, a2 from other.arr):
  a1 + a2
end
```

**AST Node:** `Expr::SFor`

**Implementation Notes:**
- Token: `TokenType::For`
- Grammar: `FOR iterator fun-header FROM expr COLON body END`
- Multiple iterator variants (map, filter, fold, etc.)

**Estimated Time:** 3-4 hours

**Test Files:**
- `test_pyret_match_for_map` ← `for map(a1 from arr): a1 + 1 end`
- `test_pyret_match_for_map2` ← `for lists.map2(a1 from self.arr, a2 from other.arr): a1 + a2 end`

---

### 4. Method Fields in Objects (1 test)

**Priority:** ⭐⭐⭐ (Important for OOP)

**Syntax:**
```pyret
{
  x: 1,
  method foo(self): self.x end,
  method _plus(self, other): self.x + other.x end
}
```

**Real Example from `test-binops.arr`:**
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

**AST Node:** `Member::SMethodField`

**Implementation Notes:**
- Extends existing object parsing
- Grammar: `METHOD name fun-header COLON body END`
- Already have data fields working, need to add method variant

**Estimated Time:** 2-3 hours

**Test Files:**
- `test_pyret_match_object_with_method` ← `{ method _plus(self, other): self.arr end }`

---

### 5. Cases Expressions (1 test)

**Priority:** ⭐⭐⭐ (Pattern matching)

**Syntax:**
```pyret
cases(Type) expr:
  | variant1(args) => result1
  | variant2(args) => result2
end
```

**Real Example from `test-binops.arr`:**
```pyret
cases(Eth.Either) run-task(thunk):
  | left(v) => "not-error"
  | right(v) => exn-unwrap(v)
end
```

**AST Node:** `Expr::SCases`

**Implementation Notes:**
- Token: `TokenType::Cases`
- Grammar: `CASES LPAREN ann RPAREN expr COLON branches END`
- Requires pattern parsing

**Estimated Time:** 4-5 hours

**Test Files:**
- `test_pyret_match_simple_cases` ← `cases(Either) e: | left(v) => v | right(v) => v end`

---

## 🎯 Priority 3: Statements and Declarations

These are top-level constructs, not expressions. May be lower priority if focusing on expression parsing.

### 6. Assignment Expressions (1 test)

**Priority:** ⭐⭐

**Syntax:**
```pyret
x := value
obj.field := new-value
```

**AST Node:** `Expr::SAssign`

**Test Files:**
- `test_pyret_match_simple_assign` ← `x := 5`

---

### 7. When Expressions (1 test)

**Priority:** ⭐⭐

**Syntax:**
```pyret
when cond: body end
```

**AST Node:** `Expr::SWhen`

**Test Files:**
- `test_pyret_match_simple_when` ← `when true: print("yes") end`

---

### 8. Let/Var Bindings (1 test)

**Priority:** ⭐⭐⭐

**Syntax:**
```pyret
x = value          # Let binding
var x = value      # Var binding (mutable)
```

**Note:** This is typically a statement, not an expression

**Test Files:**
- `test_pyret_match_simple_let` ← `x = 5`

---

### 9. Data Declarations (1 test)

**Priority:** ⭐⭐

**Syntax:**
```pyret
data Type:
  | variant1(field1, field2)
  | variant2(field3)
end
```

**Real Example from `test-lists.arr`:**
```pyret
data Box:
  | box(ref v)
end
```

**Test Files:**
- `test_pyret_match_simple_data` ← `data Box: | box(ref v) end`

---

### 10. Function Declarations (1 test)

**Priority:** ⭐⭐⭐

**Syntax:**
```pyret
fun name(params) -> return-type:
  body
end
```

**Test Files:**
- `test_pyret_match_simple_fun` ← `fun f(x): x + 1 end`

---

### 11. Import Statements (1 test)

**Priority:** ⭐

**Syntax:**
```pyret
import module as name
import file("path.arr") as name
```

**Real Example from `test-lists.arr`:**
```pyret
import equality as E
```

**Test Files:**
- `test_pyret_match_simple_import` ← `import equality as E`

---

### 12. Provide Statements (1 test)

**Priority:** ⭐

**Syntax:**
```pyret
provide *
provide { name1, name2 }
provide-types { Type1, Type2 }
```

**Real Example from `test-binops.arr`:**
```pyret
provide *
```

**Test Files:**
- `test_pyret_match_simple_provide` ← `provide *`

---

## 📈 Implementation Roadmap

### ✅ Phase 1: Core Expressions - COMPLETED!
**Goal:** Enable real Pyret code with control flow and advanced features

1. ✅ **Block expressions** (2 tests) - COMPLETED (2025-11-01 PART 1)
   - Required for control flow
   - Foundation for statements

2. ✅ **If expressions** (1 test) - COMPLETED (2025-11-01 PART 2)
   - Essential control flow
   - Used everywhere

**Status:** ✅ Completed! 69 tests passing (+2 from this phase)

### Phase 2: Advanced Features (NEXT PRIORITY)
**Goal:** Pattern matching and functional programming

3. **For expressions** (2 tests) - 3-4 hours ⭐⭐⭐ ← **NEXT!**
   - Functional list operations
   - Very Pyret-idiomatic

4. **Method fields** (1 test) - 2-3 hours ⭐⭐⭐
   - Complete object support
   - Important for OOP style

5. **Cases expressions** (1 test) - 4-5 hours ⭐⭐⭐
   - Pattern matching
   - More complex parsing

**Total:** ~9-12 hours, enables 4 new tests (33% more coverage)

### Phase 3: Statements and Declarations
**Goal:** Top-level program structure

6. **Let/Var bindings** (1 test) - 1-2 hours ⭐⭐⭐
7. **Function declarations** (1 test) - 2-3 hours ⭐⭐⭐
8. **Data declarations** (1 test) - 3-4 hours ⭐⭐
9. **Assignment** (1 test) - 1-2 hours ⭐⭐
10. **When expressions** (1 test) - 1-2 hours ⭐⭐
11. **Import/Provide** (2 tests) - 2-3 hours ⭐

**Total:** ~11-17 hours, enables 7 new tests (58% more coverage)

---

## 🧪 Test Coverage

**Current Status (2025-11-01):**
- ✅ 69 tests passing (expression-level features) ← **85.2% coverage!**
- ⏸️ 12 tests ignored (missing features)
- **Total:** 81 comparison tests

**✅ Phase 1 Complete (Blocks + If):**
- ✅ 69 tests passing (85.2% coverage)
- ⏸️ 12 tests ignored

**After Phase 2 (For + Methods + Cases):**
- 🎯 73 tests passing (90% coverage)
- ⏸️ 8 tests ignored

**After Phase 3 (All features):**
- 🎯 81 tests passing (100% coverage)
- ⏸️ 0 tests ignored

---

## 🔍 How Tests Were Generated

All tests in `comparison_tests.rs` are based on **real Pyret code** from the official repository:

1. **Source:** `/Users/jimmyhmiller/Documents/Code/open-source/pyret-lang`
2. **Files analyzed:**
   - `tests/pyret/tests/test-lists.arr` (lambdas, higher-order functions)
   - `tests/pyret/tests/test-tuple.arr` (tuple expressions)
   - `tests/pyret/tests/test-binops.arr` (operators, methods, for-expressions)
   - `examples/point.arr` (objects with methods)
   - And more...

3. **Validation:** Each test runs `./compare_parsers.sh` which:
   - Parses code with official Pyret parser → JSON
   - Parses code with our Rust parser → JSON
   - Compares ASTs for exact match

---

## 🚀 Next Steps

1. **Start with Block expressions** - highest priority, enables control flow
2. **Run tests incrementally** - use `cargo test --test comparison_tests test_pyret_match_simple_block -- --ignored`
3. **Remove `#[ignore]` attributes** as features are implemented
4. **Verify AST matches** using `./compare_parsers.sh "your-code"`

---

## 📞 Reference Files

**Test File:**
- `tests/comparison_tests.rs:554-706` - All 14 missing feature tests

**Real Pyret Examples:**
- `test-binops.arr` - Method fields, for-expressions
- `test-cases.arr` - Cases/pattern matching

**Documentation:**
- `NEXT_STEPS.md` - Implementation guides
- `src/ast.rs:292-808` - All AST node definitions
- `pyret-grammar.bnf` - Official grammar

---

**Last Updated:** 2025-11-01 (After If Expressions Implementation)
**Generated by:** Automated analysis of comparison tests against official Pyret parser
**Recent Progress:** Block expressions (PART 1) and If expressions (PART 2) completed!
