# Next Steps for Pyret Parser Implementation

**Last Updated:** 2025-11-03
**Current Status:** 110/118 tests passing (93.2%) 🎉
**Only 8 features remaining - all validated against official Pyret parser!**

---

## 🎯 PRIORITY 1: Advanced Import/Export (4 tests, ~4-6 hours)

**Why imports/exports first?**
1. ✅ Critical for real-world Pyret programs (module system)
2. ✅ 4 tests remaining (highest impact)
3. ✅ Foundation already exists (basic imports work)
4. ✅ Moderate difficulty (extend existing parsing)

The parser already handles basic imports like `import lists as L`. We need to add:

---

## 🔥 Step 1: File Imports (~1-2 hours)

### Test to Enable
```rust
#[test]
#[ignore] // TODO: File imports
fn test_import_from_file() {
    assert_matches_pyret(r#"
import file("util.arr") as U
"#);
}
```

### What This Does
File imports allow loading modules from files instead of named modules:

```pyret
import file("util.arr") as U
import file("../helpers/math.arr") as Math
```

### Implementation Guide

**1. Check Current Import Parsing**

The parser already handles `import NAME as ALIAS`. Look at the current implementation:

```rust
// In src/parser.rs, find parse_import_stmt()
fn parse_import_stmt(&mut self) -> ParseResult<Import> {
    // Currently handles: import lists as L
    // Need to add: import file("path") as Alias
}
```

**2. Check the AST Structure**

```rust
// In src/ast.rs, check Import variants
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type")]
pub enum ImportType {
    #[serde(rename = "s-import")]
    SImport {
        file: Box<Expr>,  // Can be s-const-import-type or s-str for file imports
        name: Name,
    },
    // ... other variants
}
```

**3. Parse File Import Syntax**

The import file syntax uses a function-call-like pattern:

```rust
// In src/parser.rs, update parse_import_stmt()
fn parse_import_stmt(&mut self) -> ParseResult<Import> {
    let start = self.expect(TokenType::Import)?;

    // Check for "file" keyword
    if self.matches(&TokenType::Name) && self.peek().value == "file" {
        self.advance(); // consume "file"

        // Expect opening paren
        self.expect(TokenType::LParen)?;

        // Parse file path (string literal)
        let file_path = self.parse_str()?; // Returns Expr::SStr

        // Expect closing paren
        self.expect(TokenType::RParen)?;

        // Expect "as" keyword
        self.expect(TokenType::As)?;

        // Parse module alias
        let alias = self.parse_name()?;

        // Create import with file path expression
        return Ok(Import::SImport {
            l: self.make_loc(&start, &self.previous()),
            file: Box::new(file_path),
            name: alias,
        });
    }

    // Otherwise handle regular imports (existing code)
    // ...
}
```

**4. Test It**

```bash
# Un-ignore the test
# In tests/comparison_tests.rs, remove #[ignore] from test_import_from_file

# Run the test
cargo test test_import_from_file

# Compare with official parser
./compare_parsers.sh 'import file("util.arr") as U'
```

---

## 🔧 Step 2: Provide-Types (~1-2 hours)

### Test to Enable
```rust
#[test]
#[ignore] // TODO: Provide-types
fn test_provide_with_types() {
    assert_matches_pyret(r#"
provide-types *
"#);
}
```

### What This Does
`provide-types` exports type definitions from a module:

```pyret
provide-types *              # Export all types
provide-types { List, Tree } # Export specific types
```

### Implementation Guide

**1. Check the AST Structure**

```rust
// In src/ast.rs
#[serde(rename = "s-provide-types")]
SProvideTypes {
    l: Loc,
    anns: Vec<Ann>,  // Type annotations to provide
},

#[serde(rename = "s-provide-types-all")]
SProvideTypesAll {
    l: Loc,
},
```

**2. Parse Provide-Types Statement**

```rust
// In src/parser.rs, add parse_provide_types_stmt()
fn parse_provide_types_stmt(&mut self) -> ParseResult<ProvideTypes> {
    let start = self.expect(TokenType::ProvideTypes)?;

    // Check for '*' (provide all types)
    if self.matches(&TokenType::Star) {
        self.advance();
        return Ok(ProvideTypes::SProvideTypesAll {
            l: self.make_loc(&start, &self.previous()),
        });
    }

    // Otherwise parse specific type list: { Type1, Type2 }
    self.expect(TokenType::LBrace)?;
    let types = self.parse_comma_list(|p| p.parse_ann())?;
    self.expect(TokenType::RBrace)?;

    Ok(ProvideTypes::SProvideTypes {
        l: self.make_loc(&start, &self.previous()),
        anns: types,
    })
}
```

**3. Check for ProvideTypes Token**

```bash
grep -n "ProvideTypes" src/tokenizer.rs
```

If it doesn't exist, you'll need to add it as a keyword.

**4. Test It**

```bash
cargo test test_provide_with_types
./compare_parsers.sh 'provide-types *'
```

---

## 🔧 Step 3: Provide Specific Names (~1 hour)

### Test to Enable
```rust
#[test]
#[ignore] // TODO: Provide specific names
fn test_provide_specific_names() {
    assert_matches_pyret(r#"
provide { foo, bar } end
"#);
}
```

### What This Does
Selective provide exports only specific values:

```pyret
provide { sum, product } end
provide * end  # Already works
```

### Implementation Guide

**1. Check Current Provide Parsing**

The parser already handles `provide *`. Need to add specific name lists.

**2. Update Provide Parsing**

```rust
// In src/parser.rs, update parse_provide_stmt()
fn parse_provide_stmt(&mut self) -> ParseResult<Provide> {
    let start = self.expect(TokenType::Provide)?;

    // Check for '*' (provide all)
    if self.matches(&TokenType::Star) {
        self.advance();
        self.expect(TokenType::End)?;
        return Ok(Provide::SProvideAll {
            l: self.make_loc(&start, &self.previous()),
        });
    }

    // Parse specific names: { name1, name2 }
    self.expect(TokenType::LBrace)?;
    let names = self.parse_comma_list(|p| p.parse_name())?;
    self.expect(TokenType::RBrace)?;
    self.expect(TokenType::End)?;

    Ok(Provide::SProvide {
        l: self.make_loc(&start, &self.previous()),
        names,
    })
}
```

---

## 🔧 Step 4: Realistic Module Structure (~1-2 hours)

### Test to Enable
```rust
#[test]
#[ignore] // TODO: Complex module structure
fn test_realistic_module_structure() {
    assert_matches_pyret(r#"
import file("../util.arr") as U
import lists as L
provide { process, validate } end
provide-types { Result }

fun process(x): U.helper(x) end
fun validate(y): L.filter(y, lam(v): v > 0 end) end
"#);
}
```

### What This Does
Tests multiple imports and exports together in a realistic module.

### Implementation Guide

This test should work automatically once the previous 3 features are implemented. Just:
1. Enable the test
2. Run it
3. Debug any issues with interaction between features

---

## ✅ Step 5: Verify All Import/Export Tests Pass

After implementing all four features:

```bash
# Run all comparison tests
cargo test --test comparison_tests

# Should see 114/118 passing (was 110/118)
# Remaining: 4 tests (object extension, tables, spy)
```

---

## 📋 Summary: Import/Export Implementation Checklist

### Phase 1: File Imports (~1-2 hours)
- [ ] Update `parse_import_stmt()` to handle `import file("path") as Alias`
- [ ] Parse `file` keyword, parentheses, string path, `as`, and alias
- [ ] Create `Import::SImport` with file path expression
- [ ] Un-ignore `test_import_from_file`
- [ ] Test: `./compare_parsers.sh 'import file("util.arr") as U'`

### Phase 2: Provide-Types (~1-2 hours)
- [ ] Check/add `ProvideTypes` token to tokenizer
- [ ] Implement `parse_provide_types_stmt()`
- [ ] Handle `provide-types *` (all types)
- [ ] Handle `provide-types { Type1, Type2 }` (specific types)
- [ ] Un-ignore `test_provide_with_types`
- [ ] Test: `./compare_parsers.sh 'provide-types *'`

### Phase 3: Provide Specific Names (~1 hour)
- [ ] Update `parse_provide_stmt()` to handle `provide { name1, name2 } end`
- [ ] Parse brace-enclosed name list
- [ ] Un-ignore `test_provide_specific_names`
- [ ] Test: `./compare_parsers.sh 'provide { foo, bar } end'`

### Phase 4: Complex Module Structure (~1-2 hours)
- [ ] Un-ignore `test_realistic_module_structure`
- [ ] Test multiple imports and exports together
- [ ] Debug any interaction issues

### Final Verification
- [ ] Run all tests: `cargo test`
- [ ] Should have 114/118 passing (96.6%)
- [ ] Update CLAUDE.md with new status
- [ ] Celebrate! 🎉

---

## 🎯 After Import/Export: What's Next? (4 Remaining Tests)

Once import/export is done, the remaining features are:

1. **Object Extension** (2 tests, ~3-4 hours)
   - `point.{ z: 0 }` syntax
   - AST nodes already exist: `SExtend`, `SUpdate`

2. **Table Literals** (1 test, ~4-6 hours)
   - `table: name, age row: "Alice", 30 end`
   - Most complex remaining feature

3. **Spy Expressions** (1 test, ~1-2 hours)
   - `spy: x end` (may already work, needs investigation)

---

## 💡 Tips for Success

1. **Test incrementally** - Run tests after each small change
2. **Use the comparison script** - `./compare_parsers.sh` is your friend
3. **Check existing code** - AST nodes already exist for most features!
4. **Follow patterns** - Look at similar parsing code (e.g., regular imports for file imports)
5. **Check the grammar** - `/Users/jimmyhmiller/Documents/Code/open-source/pyret-lang/src/js/base/pyret-grammar.bnf`

**Start with file imports - it's a natural extension of existing import parsing!** ✨
