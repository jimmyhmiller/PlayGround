# Comprehensive Pyret Tokenizer vs Pest Grammar Comparison Report

**Date:** 2025-10-31
**Source Files:**
- JavaScript Tokenizer: `/Users/jimmyhmiller/Documents/Code/open-source/pyret-lang/src/js/base/pyret-tokenizer.js`
- Pest Grammar: `/Users/jimmyhmiller/Documents/Code/PlayGround/claude-experiments/pyret-attempt/src/pyret.pest`

---

## Executive Summary

**Total Analysis:**
- JavaScript tokens: **120**
- Pest tokens: **127**
- ✅ Correct matches: **109** (90.8%)
- ❌ Mismatches: **8** (6.7%)
- ⚠️ Missing in Pest: **3** (2.5%)
- ➕ Extra in Pest: **10** (intentional additions)

**Critical Finding:** The Pest grammar has **6 major table operation token mismatches** and **2 other token mismatches** that will prevent parsing of valid Pyret code.

---

## Critical Issues (High Priority)

### 1. Table Operations - ALL INCORRECT ❌❌❌

The JavaScript tokenizer uses **simple keywords** for table operations, but the Pest grammar expects **compound keywords**. This is a fundamental error that will break all table-related code.

| Token Name | JavaScript Value | Pest Value | Actual Pyret Syntax |
|------------|-----------------|------------|---------------------|
| `TABLE-SELECT` / `TABLE_SELECT` | `"select"` | `"table-select"` | `select ... from ... end` |
| `TABLE-FILTER` / `TABLE_FILTER` | `"sieve"` | `"table-filter"` | `sieve ... using ... end` |
| `TABLE-ORDER` / `TABLE_ORDER` | `"order"` | `"table-order"` | `order ... : ... end` |
| `TABLE-EXTEND` / `TABLE_EXTEND` | `"extend"` | `"table-extend"` | `extend ... using ... end` |
| `TABLE-EXTRACT` / `TABLE_EXTRACT` | `"extract"` | `"table-extract"` | `extract ... from ... end` |
| `TABLE-UPDATE` / `TABLE_UPDATE` | `"transform"` | `"table-update"` | `transform ... using ... end` |

**Example of the problem:**
```pyret
# Valid Pyret code:
filtered = sieve employees using age:
  age > 18
end

# What Pest currently expects (WRONG):
filtered = table-filter employees using age:
  age > 18
end
```

**Required Fix:**
```pest
# BEFORE (WRONG):
TABLE_SELECT = @{ "table-select" ~ kw_boundary }
TABLE_FILTER = @{ "table-filter" ~ kw_boundary }
TABLE_ORDER = @{ "table-order" ~ kw_boundary }
TABLE_EXTEND = @{ "table-extend" ~ kw_boundary }
TABLE_EXTRACT = @{ "table-extract" ~ kw_boundary }
TABLE_UPDATE = @{ "table-update" ~ kw_boundary }

# AFTER (CORRECT):
TABLE_SELECT = @{ "select" ~ kw_boundary }
TABLE_FILTER = @{ "sieve" ~ kw_boundary }
TABLE_ORDER = @{ "order" ~ kw_boundary }
TABLE_EXTEND = @{ "extend" ~ kw_boundary }
TABLE_EXTRACT = @{ "extract" ~ kw_boundary }
TABLE_UPDATE = @{ "transform" ~ kw_boundary }
```

---

### 2. Other Token Mismatches

#### REACTOR Token ❌
- **JavaScript:** `"reactor"` (no colon)
- **Pest:** `"reactor:"` (with colon)
- **Fix:** Change Pest from `"reactor:"` to `"reactor"`

**Note:** The BNF grammar shows `reactor-expr: REACTOR COLON`, meaning REACTOR is followed by a separate COLON token, not included in the reactor keyword itself.

#### LOAD-TABLE Token ⚠️
- **JavaScript:** `"load-table"` (no colon)
- **Pest:** `"load-table:"` (with colon)
- **Fix:** Change Pest from `"load-table:"` to `"load-table"`

**Note:** Similar to REACTOR, the BNF shows `load-table-expr: LOAD-TABLE COLON`, indicating COLON is separate.

---

## Missing Tokens (Medium Priority)

Three tokens are defined in the JavaScript tokenizer but missing from Pest:

### 1. BACKSLASH ⚠️
- **Value:** `"\\"`
- **Add to Pest:**
```pest
BACKSLASH = { "\\" }
```

### 2. BY ⚠️
- **Value:** `"by"`
- **Add to Pest:**
```pest
BY = @{ "by" ~ kw_boundary }
```

### 3. DO ⚠️
- **Value:** `"do"`
- **Add to Pest:**
```pest
DO = @{ "do" ~ kw_boundary }
```

---

## Extra Tokens in Pest (Informational)

These tokens exist in Pest but not in the JavaScript tokenizer. Most are intentional:

### Pest-Specific Tokens (OK ✓)
These are intentional additions for Pest's parsing needs:
- `PARENSPACE`, `PARENNOSPACE`, `PARENAFTERBRACE` - Different paren contexts
- `BAD_NUMBER`, `BAD_OPER`, `UNKNOWN` - Error tokens
- `UNTERMINATED_BLOCK_COMMENT` - Error token

### Incorrectly Extracted Values (⚠️)
These appear to be extraction errors in my analysis script (they're pattern rules, not literal tokens):
- `NUMBER: "-"` (should be a pattern, not a literal)
- `RATIONAL: "-"` (should be a pattern, not a literal)
- `ROUGHRATIONAL: "~"` (should be a pattern, not a literal)

**Note:** These are fine in the actual Pest file - they're defined as complex regex patterns, not simple literals.

---

## Correctly Matched Tokens ✅

109 tokens match perfectly between JavaScript and Pest. Here's a sample:

| Token | Value | Status |
|-------|-------|--------|
| `AND` | `"and"` | ✅ |
| `AS` | `"as"` | ✅ |
| `BLOCK` | `"block:"` | ✅ |
| `CHECK` | `"check"` | ✅ |
| `CHECKCOLON` | `"check:"` | ✅ |
| `COLON` | `":"` | ✅ |
| `COLONCOLON` | `"::"` | ✅ |
| `DATA` | `"data"` | ✅ |
| `DOC` | `"doc:"` | ✅ |
| `ELSE` | `"else"` | ✅ |
| `ELSECOLON` | `"else:"` | ✅ |
| `END` | `"end"` | ✅ |
| `FUN` | `"fun"` | ✅ |
| `IF` | `"if"` | ✅ |
| `IS` | `"is"` | ✅ |
| `ISNOT` | `"is-not"` | ✅ |
| `LAM` | `"lam"` | ✅ |
| `LET` | `"let"` | ✅ |
| `METHOD` | `"method"` | ✅ |
| `WHERE` | `"where:"` | ✅ |

... and 89 more tokens match correctly.

---

## Pattern Analysis

### Tokens with Colons
- **JavaScript:** 17 tokens ending with `:`
- **Pest:** 19 tokens ending with `:`
- **Mismatches:** 2 tokens (REACTOR, LOAD-TABLE)

**Pattern:** Most colon tokens match correctly. The two mismatches occur where Pest incorrectly adds a colon that should be a separate token.

### Hyphenated Keywords
- Tokens like `is-not`, `is-roughly`, `else if`, `type-let` all match correctly
- The table operations are the only hyphenated keywords with mismatches

---

## Recommended Action Plan

### Phase 1: Critical Fixes (DO THIS FIRST)
1. **Fix table operation tokens in pyret.pest:**
   ```pest
   TABLE_SELECT = @{ "select" ~ kw_boundary }
   TABLE_FILTER = @{ "sieve" ~ kw_boundary }
   TABLE_ORDER = @{ "order" ~ kw_boundary }
   TABLE_EXTEND = @{ "extend" ~ kw_boundary }
   TABLE_EXTRACT = @{ "extract" ~ kw_boundary }
   TABLE_UPDATE = @{ "transform" ~ kw_boundary }
   ```

2. **Fix REACTOR token:**
   ```pest
   REACTOR = @{ "reactor" ~ kw_boundary }
   ```

3. **Fix LOAD_TABLE token:**
   ```pest
   LOAD_TABLE = @{ "load-table" ~ kw_boundary }
   ```

### Phase 2: Add Missing Tokens
Add the three missing tokens:
```pest
BACKSLASH = { "\\" }
BY = @{ "by" ~ kw_boundary }
DO = @{ "do" ~ kw_boundary }
```

And add them to the KEYWORDS list:
```pest
KEYWORDS = _{
    // ... existing keywords ...
    BACKSLASH | BY | DO
}
```

### Phase 3: Testing
Create test cases for:
1. Each table operation (`select`, `sieve`, `order`, `extend`, `extract`, `transform`)
2. Reactor expressions
3. Load-table expressions
4. Backslash, by, and do keywords in their appropriate contexts

---

## Verification Commands

After making fixes, verify with:

```bash
# Test table operations
echo 'select name from employees end' | your_parser
echo 'sieve table using x: x > 5 end' | your_parser
echo 'order table: name ascending end' | your_parser

# Test reactor
echo 'reactor: on-tick: update-state end' | your_parser

# Test load-table
echo 'load-table: name :: String source: "data.csv" end' | your_parser
```

---

## References

- Pyret Official Documentation: https://pyret.org/docs/latest/tables.html
- Pyret Grammar BNF: `/Users/jimmyhmiller/Documents/Code/open-source/pyret-lang/src/js/base/pyret-grammar.bnf`
- Pyret Tokenizer: `/Users/jimmyhmiller/Documents/Code/open-source/pyret-lang/src/js/base/pyret-tokenizer.js`

---

## Conclusion

The Pest grammar has **8 critical token mismatches** that must be fixed:
- 6 table operations use wrong keywords
- 2 tokens (REACTOR, LOAD_TABLE) incorrectly include colons

Additionally, 3 tokens (BACKSLASH, BY, DO) are missing and should be added.

Once these fixes are applied, the grammar will correctly match 120/120 tokens from the JavaScript tokenizer (100% compatibility).
