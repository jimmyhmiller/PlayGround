# Next Steps for Pyret Parser Implementation

**Last Updated:** 2025-01-30
**Current Status:** ✅ Functions and When expressions complete! Ready for remaining features.
**Tests Passing:** 68/68 parser tests ✅ (100%), 76/81 comparison tests ✅ (93.8%)

---

## ✅ COMPLETED FEATURES

### Method Fields in Objects ✅
Method fields are now fully working!

**What was completed:**
1. ✅ Implemented `parse_method_field()` - Parses method syntax in objects
2. ✅ Method parameter parsing with `Bind` structures
3. ✅ Correctly distinguishes `params` (type parameters) from `args` (function parameters)
4. ✅ Optional return type annotation support (`-> ann`)
5. ✅ Optional where clause support for tests
6. ✅ Added JSON serialization for `SMethodField` with correct field ordering
7. ✅ Added comprehensive parser test `test_parse_object_with_method`
8. ✅ Enabled comparison test `test_pyret_match_object_with_method`

### Function Definitions ✅
Function definitions (`fun f(x): ... end`) are now fully working!

**What was completed:**
1. ✅ Implemented `parse_fun_expr()` - Parses function definitions
2. ✅ Function name parsing
3. ✅ Parameter parsing with `Bind` structures
4. ✅ Optional return type annotations
5. ✅ Optional where clause support
6. ✅ Added JSON serialization for `SFun`
7. ✅ Enabled comparison test `test_pyret_match_simple_fun`

### When Expressions ✅
When expressions (`when cond: ... end`) are now fully working!

**What was completed:**
1. ✅ Implemented `parse_when_expr()` - Parses when expressions
2. ✅ Condition expression parsing
3. ✅ Body block parsing
4. ✅ Added JSON serialization for `SWhen`

All 76 passing comparison tests produce identical ASTs to the official Pyret parser!

---

## 📋 Next Priority Tasks (IN ORDER)

### 1. parse_data_expr() - Data Definitions ⭐⭐⭐⭐ (HIGHEST PRIORITY)
**Why:** Required for 1 comparison test (`simple_data`)

**Grammar:**
```bnf
data-expr: DATA NAME ty-params data-mixins COLON first-data-variant data-variant* data-sharing where-clause END
data-variant: BAR NAME variant-members data-with | BAR NAME variant-members
variant-members: (PARENNOSPACE|PARENAFTERBRACE) [variant-member (COMMA variant-member)*] RPAREN
variant-member: [REF] binding
data-with: WITH fields END
data-sharing: [SHARING fields]
```

**Example:**
```pyret
data Box:
  | box(ref v)
end
```

**AST Node:** `Expr::SData { l, name, params, mixins, variants, shared_members, check_loc, check }`

**Implementation Steps:**

1. **Study the SData AST in `src/ast.rs`:**
   - Look at the `Expr::SData` variant
   - Look at the `Variant` and `VariantMember` structs
   - Understand how data definitions are structured

2. **Add DATA case to parse_prim_expr():**
   ```rust
   TokenType::Data => self.parse_data_expr(),
   ```

3. **Implement parse_data_expr():**
   - Parse `data` keyword
   - Parse data type name
   - Parse type parameters (empty for now)
   - Parse variants (e.g., `| box(ref v)`)
   - Parse optional sharing clause
   - Parse optional where clause
   - Build `Expr::SData` node

4. **Add JSON serialization in to_pyret_json.rs:**
   - Look at how variants are serialized
   - Match the official Pyret JSON format

5. **Enable comparison test:**
   - Remove `#[ignore]` from `test_pyret_match_simple_data`

**Estimated Time:** 3-4 hours

---

### 2. parse_cases_expr() - Cases Expressions ⭐⭐⭐
**Why:** Required for 1 comparison test (`simple_cases`)

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

**Implementation Steps:**

1. **Study the SCases AST**
2. **Add CASES case to parse_prim_expr()**
3. **Implement parse_cases_expr():**
   - Parse `cases` keyword and type annotation
   - Parse value expression
   - Parse branches (pattern => body)
   - Parse optional else clause
4. **Add JSON serialization**
5. **Enable comparison test**

**Estimated Time:** 4-5 hours

---

### 3. parse_assign_expr() - Assignment Expressions ⭐⭐
**Why:** Required for remaining comparison tests

**Grammar:**
```bnf
assign-expr: id COLONEQUALS expr
```

**Example:** `x := 5`

**AST Node:** `Expr::SAssign { l, id, value }`

**Implementation Steps:**

1. **Study the SAssign AST**
2. **Add to parse_binop_expr() or parse_expr():**
   - Check for `:=` operator after parsing id
3. **Implement parse_assign_expr():**
   - Parse identifier
   - Expect `:=` token
   - Parse right-hand side expression
4. **Add JSON serialization**
5. **Add tests**

**Estimated Time:** 1-2 hours

---

### 4. parse_import_expr() - Import Statements ⭐
**Why:** Required for 1 comparison test (`simple_import`)

**Example:** `import file("foo.arr") as F`

**Estimated Time:** 2-3 hours

---

### 5. parse_provide_expr() - Provide Statements ⭐
**Why:** Required for 1 comparison test (`simple_provide`)

**Example:** `provide x, y end`

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
- ✅ 76/81 comparison tests passing (93.8%)
- ✅ Functions, method fields, and when expressions complete - all ASTs match Pyret parser
- ✅ 68/68 parser tests passing (100%)

**Next Feature: DATA DEFINITIONS**

Data definitions are the next highest priority feature. They allow defining algebraic data types with variants:

```pyret
data Box:
  | box(ref v)
end
```

**What to do:**
1. Study the `Expr::SData` variant in `src/ast.rs`
2. Study the `Variant` and `VariantMember` structs
3. Look at the Pyret grammar for data definitions
4. Add `TokenType::Data => self.parse_data_expr()` to `parse_prim_expr()`
5. Implement `parse_data_expr()`:
   - Parse data name
   - Parse variants (one or more, starting with `|`)
   - Parse variant members (parameters)
   - Parse optional sharing clause
   - Parse optional where clause
6. Add JSON serialization for `SData` in `to_pyret_json.rs`
7. Enable comparison test by removing `#[ignore]` from `test_pyret_match_simple_data`
8. Run `./compare_parsers.sh "data Box: | box(ref v) end"`

**Estimated Time:** 3-4 hours

**We're at 93.8% completion!** Only 5 more features to go! 🚀

**Remaining features:**
1. Data definitions (1 test) ⭐⭐⭐⭐
2. Cases expressions (1 test) ⭐⭐⭐
3. Assignment expressions ⭐⭐
4. Import statements (1 test) ⭐
5. Provide statements (1 test) ⭐

