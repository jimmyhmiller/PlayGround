# Pyret Parser Implementation Plan

## 📊 Project Status

**Current Phase:** Phase 3 - Expressions (IN PROGRESS)
**Last Updated:** 2025-10-31
**Overall Progress:** 35% (Phases 1-2 Complete, Phase 3 Partial)

| Phase | Status | Progress |
|-------|--------|----------|
| Phase 1: Foundation | ✅ Complete | 100% |
| Phase 2: Parser Core | ✅ Complete | 100% |
| Phase 3: Expressions | 🚧 In Progress | 35% |
| Phase 4: Control Flow | ⏳ Pending | 0% |
| Phase 5: Functions & Bindings | ⏳ Pending | 0% |
| Phase 6: Data Definitions | ⏳ Pending | 0% |
| Phase 7: Type System | ⏳ Pending | 0% |
| Phase 8: Imports/Exports | ⏳ Pending | 0% |
| Phase 9: Tables | ⏳ Pending | 0% |
| Phase 10: Testing & Reactors | ⏳ Pending | 0% |
| Phase 11: Program Structure | ⏳ Pending | 0% |
| Phase 12: Polish | ⏳ Pending | 0% |

**Quick Links:**
- ✅ [Phase 1 Complete Summary](PHASE1_COMPLETE.md)
- ✅ [Phase 2 Complete Summary](PHASE2_COMPLETE.md)
- ✅ [Phase 3 Partial Complete (Parens & Apps)](PHASE3_PARENS_AND_APPS_COMPLETE.md)
- 📝 [Implementation Order](#implementation-order) (see below)
- 🎯 [Next Steps](#phase-3-expressions-continue) (Phase 3 continuation)

---

## Overview
Create a comprehensive Pyret parser in Rust that generates JSON AST matching the reference JavaScript implementation exactly. The parser will be hand-written recursive descent, processing ~302 BNF rules into ~150 distinct AST node types.

## File Changes

### 1. Update CLAUDE.md
**File:** `/Users/jimmyhmiller/Documents/Code/PlayGround/CLAUDE.md`
- Add new section documenting the Pyret parser project
- Include architecture overview, structure, usage examples, and development commands
- Document all 150+ AST node types organized by category
- Add testing strategy and reference file locations
- Document the single-file approach for both AST and parser

### 2. Create Single AST File
**File:** `src/ast.rs` (~3000-3500 lines)
- **All AST node types in one well-organized file**
- Organized in clear sections with documentation comments:
  - Section 1: Location types (`Loc`)
  - Section 2: Names (6 variants)
  - Section 3: Annotations (12 variants)
  - Section 4: Bindings
  - Section 5: Expressions (60+ variants)
  - Section 6: Members & Fields
  - Section 7: Variants & Data
  - Section 8: Branches (if/cases)
  - Section 9: Imports/Exports
  - Section 10: Table Operations
  - Section 11: Check Operations
  - Section 12: Top-level Program
- All types with serde serialization configured for exact JSON match

### 3. Create Single Parser File
**File:** `src/parser.rs` (~2500-3000 lines)
- **All parser logic in one well-organized file**
- Organized in clear sections matching AST structure:
  - Section 1: Parser struct and core methods (peek, advance, expect, matches, etc.)
  - Section 2: Program & top-level (parse_program, parse_prelude, parse_block)
  - Section 3: Import/Export parsing
  - Section 4: Type annotation parsing (parse_ann, parse_arrow_ann, parse_record_ann, etc.)
  - Section 5: Binding parsing (parse_bind, parse_let_bind, parse_tuple_bind)
  - Section 6: Expression parsing (parse_expr, parse_binop_expr, parse_prim_expr)
  - Section 7: Control flow (parse_if_expr, parse_cases_expr, parse_when_expr, parse_for_expr)
  - Section 8: Functions (parse_fun_expr, parse_lambda_expr, parse_method_expr)
  - Section 9: Data definitions (parse_data_expr, parse_variant, parse_data_with)
  - Section 10: Table operations (parse_table_expr, parse_table_select, etc.)
  - Section 11: Check/test parsing (parse_check_expr, parse_check_test)
  - Section 12: Helper methods (parse_comma_list, parse_optional, etc.)
- Direct 1-to-1 mapping from BNF rules to Rust functions
- Comprehensive inline documentation

### 4. Create Error Handling
**File:** `src/error.rs` (~150 lines)
- Define `ParseError` type with source location
- Implement Display/Error traits
- Include context about what was being parsed
- Helper functions for common error patterns

### 5. Update lib.rs
**File:** `src/lib.rs` (~50 lines)
- Add module declarations: `pub mod ast;`, `pub mod parser;`, `pub mod error;`, `pub mod tokenizer;`
- Export public types: `pub use ast::*;`, `pub use parser::Parser;`, `pub use error::ParseError;`
- Add documentation for the library

### 6. Update Cargo.toml
**File:** `Cargo.toml`
- Add dependencies:
  - `serde = { version = "1.0", features = ["derive"] }`
  - `serde_json = "1.0"`
  - `thiserror = "1.0"`
  - `anyhow = "1.0"`
- Add dev-dependencies:
  - `pretty_assertions = "1.4"`
  - `insta = "1.34"` (snapshot testing for JSON comparison)

### 7. Create Test Files
**Directory:** `tests/`
- `parser_tests.rs` - Comprehensive unit tests for parser functions
- `integration_tests.rs` - Full program parsing tests
- `json_comparison_tests.rs` - Compare output against reference using `node ast-to-json.jarr`
- `fixtures/*.arr` - Sample Pyret files for testing

### 8. Update main.rs
**File:** `src/main.rs` (~100-150 lines)
- Add CLI for parsing Pyret files and outputting JSON
- Example usage showing: tokenizer → parser → JSON output
- Add `--compare` flag to compare with reference implementation
- Add `--pretty` flag for pretty-printed JSON
- Example:
  ```rust
  // Parse file
  let tokens = Tokenizer::new(source, filename).tokenize()?;
  let mut parser = Parser::new(tokens, filename);
  let ast = parser.parse_program()?;
  let json = serde_json::to_string_pretty(&ast)?;
  println!("{}", json);
  ```

## File Structure Summary

```
pyret-attempt2/
├── Cargo.toml               # Updated with dependencies
├── PARSER_PLAN.md           # This document
├── src/
│   ├── main.rs              # CLI tool (~150 lines)
│   ├── lib.rs               # Library root (~50 lines)
│   ├── tokenizer.rs         # [EXISTING] Tokenizer
│   ├── ast.rs               # ALL AST types (~3500 lines)
│   ├── parser.rs            # ALL parser logic (~3000 lines)
│   └── error.rs             # Error types (~150 lines)
└── tests/
    ├── parser_tests.rs
    ├── integration_tests.rs
    ├── json_comparison_tests.rs
    └── fixtures/
        └── *.arr            # Test Pyret files
```

## ast.rs Organization (~3500 lines)

```rust
//! Pyret Abstract Syntax Tree
//!
//! Complete AST node definitions matching the reference implementation.
//! All nodes are serializable to JSON with exact format matching.

use serde::{Serialize, Serializer};

// ============================================================================
// SECTION 1: Source Locations (~50 lines)
// ============================================================================

#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Loc {
    pub source: String,
    #[serde(rename = "start-line")]
    pub start_line: u32,
    #[serde(rename = "start-column")]
    pub start_column: u32,
    #[serde(rename = "start-char")]
    pub start_char: u32,
    #[serde(rename = "end-line")]
    pub end_line: u32,
    #[serde(rename = "end-column")]
    pub end_column: u32,
    #[serde(rename = "end-char")]
    pub end_char: u32,
}

// ============================================================================
// SECTION 2: Names (6 variants, ~80 lines)
// ============================================================================

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type")]
pub enum Name {
    #[serde(rename = "s-underscore")]
    SUnder score { l: Loc },

    #[serde(rename = "s-name")]
    SName { l: Loc, s: String },

    #[serde(rename = "s-global")]
    SGlobal { s: String },

    #[serde(rename = "s-module-global")]
    SModuleGlobal { s: String },

    #[serde(rename = "s-type-global")]
    STypeGlobal { s: String },

    #[serde(rename = "s-atom")]
    SAtom { base: String, serial: u32 },
}

// ============================================================================
// SECTION 3: Type Annotations (12 variants, ~200 lines)
// ============================================================================

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type")]
pub enum Ann {
    #[serde(rename = "a-blank")]
    ABlank,

    #[serde(rename = "a-any")]
    AAny { l: Loc },

    #[serde(rename = "a-name")]
    AName { l: Loc, id: Name },

    #[serde(rename = "a-type-var")]
    ATypeVar { l: Loc, id: Name },

    #[serde(rename = "a-arrow")]
    AArrow {
        l: Loc,
        args: Vec<Ann>,
        ret: Box<Ann>,
        #[serde(rename = "use-parens")]
        use_parens: bool,
    },

    #[serde(rename = "a-arrow-argnames")]
    AArrowArgnames {
        l: Loc,
        args: Vec<AField>,
        ret: Box<Ann>,
        #[serde(rename = "use-parens")]
        use_parens: bool,
    },

    #[serde(rename = "a-method")]
    AMethod { l: Loc, args: Vec<Ann>, ret: Box<Ann> },

    #[serde(rename = "a-record")]
    ARecord { l: Loc, fields: Vec<AField> },

    #[serde(rename = "a-tuple")]
    ATuple { l: Loc, fields: Vec<Ann> },

    #[serde(rename = "a-app")]
    AApp { l: Loc, ann: Box<Ann>, args: Vec<Ann> },

    #[serde(rename = "a-pred")]
    APred { l: Loc, ann: Box<Ann>, exp: Box<Expr> },

    #[serde(rename = "a-dot")]
    ADot { l: Loc, obj: Name, field: String },
}

// ============================================================================
// SECTION 4: Bindings (~150 lines)
// ============================================================================

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type")]
pub enum Bind {
    #[serde(rename = "s-bind")]
    SBind {
        l: Loc,
        shadows: bool,
        id: Name,
        ann: Ann,
    },

    #[serde(rename = "s-tuple-bind")]
    STupleBind {
        l: Loc,
        fields: Vec<Bind>,
        #[serde(rename = "as-name")]
        as_name: Option<Box<Bind>>,
    },
}

// ============================================================================
// SECTION 5: Expressions (60+ variants, ~1500 lines)
// ============================================================================

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type")]
pub enum Expr {
    // Control flow
    #[serde(rename = "s-if")]
    SIf { l: Loc, branches: Vec<IfBranch>, blocky: bool },

    #[serde(rename = "s-if-else")]
    SIfElse {
        l: Loc,
        branches: Vec<IfBranch>,
        #[serde(rename = "_else")]
        _else: Box<Expr>,
        blocky: bool
    },

    // ... 58+ more expression variants

    // Primitives
    #[serde(rename = "s-num")]
    SNum { l: Loc, n: f64 },

    #[serde(rename = "s-str")]
    SStr { l: Loc, s: String },

    #[serde(rename = "s-bool")]
    SBool { l: Loc, b: bool },

    // ... etc
}

// ============================================================================
// SECTION 6: Members & Fields (~100 lines)
// ============================================================================

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type")]
pub enum Member {
    #[serde(rename = "s-data-field")]
    SDataField { l: Loc, name: String, value: Box<Expr> },

    #[serde(rename = "s-mutable-field")]
    SMutableField { l: Loc, name: String, ann: Ann, value: Box<Expr> },

    #[serde(rename = "s-method-field")]
    SMethodField {
        l: Loc,
        name: String,
        params: Vec<Name>,
        args: Vec<Bind>,
        ann: Ann,
        doc: String,
        body: Box<Expr>,
        #[serde(rename = "_check-loc")]
        check_loc: Option<Loc>,
        #[serde(rename = "_check")]
        check: Option<Box<Expr>>,
        blocky: bool,
    },
}

// ============================================================================
// SECTION 7: Variants & Data (~150 lines)
// ============================================================================
// ... data definition types

// ============================================================================
// SECTION 8: Branches (if/cases) (~150 lines)
// ============================================================================
// ... branch types

// ============================================================================
// SECTION 9: Imports/Exports (~300 lines)
// ============================================================================
// ... import/export types

// ============================================================================
// SECTION 10: Table Operations (~200 lines)
// ============================================================================
// ... table types

// ============================================================================
// SECTION 11: Check Operations (~150 lines)
// ============================================================================
// ... check/test types

// ============================================================================
// SECTION 12: Top-level Program (~100 lines)
// ============================================================================

#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "type")]
pub struct Program {
    #[serde(rename = "type")]
    pub node_type: String, // "s-program"
    pub l: Loc,
    #[serde(rename = "_use")]
    pub use_stmt: Option<Use>,
    #[serde(rename = "_provide")]
    pub provide: Provide,
    #[serde(rename = "provided-types")]
    pub provided_types: ProvideTypes,
    pub provides: Vec<ProvideBlock>,
    pub imports: Vec<Import>,
    pub block: Box<Expr>,
}
```

## parser.rs Organization (~3000 lines)

```rust
//! Pyret Parser
//!
//! Hand-written recursive descent parser implementing the complete Pyret grammar.
//! Each BNF rule maps to a corresponding parse function.

use crate::ast::*;
use crate::error::ParseError;
use crate::tokenizer::{Token, TokenType};

// ============================================================================
// SECTION 1: Parser Struct and Core Methods (~200 lines)
// ============================================================================

pub struct Parser {
    tokens: Vec<Token>,
    current: usize,
    file_name: String,
}

impl Parser {
    pub fn new(tokens: Vec<Token>, file_name: String) -> Self {
        Parser {
            tokens,
            current: 0,
            file_name,
        }
    }

    // Core navigation
    fn peek(&self) -> &Token {
        &self.tokens[self.current]
    }

    fn advance(&mut self) -> &Token {
        if !self.is_at_end() {
            self.current += 1;
        }
        &self.tokens[self.current - 1]
    }

    fn expect(&mut self, token_type: TokenType) -> Result<Token, ParseError> {
        if self.matches(token_type) {
            Ok(self.advance().clone())
        } else {
            Err(ParseError::expected(token_type, self.peek().clone()))
        }
    }

    fn matches(&self, token_type: TokenType) -> bool {
        !self.is_at_end() && self.peek().token_type == token_type
    }

    fn is_at_end(&self) -> bool {
        self.peek().token_type == TokenType::Eof
    }

    // Location helpers
    fn current_loc(&self) -> Loc {
        let token = self.peek();
        token.location.clone()
    }

    fn make_loc(&self, start_token: &Token, end_token: &Token) -> Loc {
        Loc {
            source: self.file_name.clone(),
            start_line: start_token.location.start_line,
            start_col: start_token.location.start_col,
            start_pos: start_token.location.start_pos,
            end_line: end_token.location.end_line,
            end_col: end_token.location.end_col,
            end_pos: end_token.location.end_pos,
        }
    }
}

// ============================================================================
// SECTION 2: Program & Top-Level (~150 lines)
// ============================================================================

impl Parser {
    /// program: prelude block
    pub fn parse_program(&mut self) -> Result<Program, ParseError> {
        let start = self.peek().clone();

        // Parse prelude (use, imports, provides)
        let prelude = self.parse_prelude()?;

        // Parse main block
        let block = self.parse_block()?;

        let end = self.tokens[self.current - 1].clone();
        let loc = self.make_loc(&start, &end);

        Ok(Program {
            node_type: "s-program".to_string(),
            l: loc,
            use_stmt: prelude.use_stmt,
            provide: prelude.provide,
            provided_types: prelude.provided_types,
            provides: prelude.provides,
            imports: prelude.imports,
            block: Box::new(block),
        })
    }

    /// prelude: [use-stmt] (provide-stmt|import-stmt)*
    fn parse_prelude(&mut self) -> Result<Prelude, ParseError> {
        // TODO: implement
        todo!()
    }

    /// block: stmt*
    fn parse_block(&mut self) -> Result<Expr, ParseError> {
        // TODO: implement
        todo!()
    }
}

// ============================================================================
// SECTION 3: Import/Export Parsing (~300 lines)
// ============================================================================

impl Parser {
    /// use-stmt: USE NAME import-source
    fn parse_use_stmt(&mut self) -> Result<Use, ParseError> {
        // TODO: implement
        todo!()
    }

    /// import-stmt: INCLUDE | IMPORT | ...
    fn parse_import_stmt(&mut self) -> Result<Import, ParseError> {
        // TODO: implement
        todo!()
    }

    /// provide-stmt: PROVIDE stmt END | ...
    fn parse_provide_stmt(&mut self) -> Result<Provide, ParseError> {
        // TODO: implement
        todo!()
    }

    // ... all import/export rules
}

// ============================================================================
// SECTION 4: Type Annotation Parsing (~250 lines)
// ============================================================================

impl Parser {
    /// ann: name-ann | record-ann | arrow-ann | ...
    fn parse_ann(&mut self) -> Result<Ann, ParseError> {
        // TODO: implement
        todo!()
    }

    /// arrow-ann: (args -> ret)
    fn parse_arrow_ann(&mut self) -> Result<Ann, ParseError> {
        // TODO: implement
        todo!()
    }

    /// record-ann: { field, ... }
    fn parse_record_ann(&mut self) -> Result<Ann, ParseError> {
        // TODO: implement
        todo!()
    }

    // ... all annotation rules
}

// ============================================================================
// SECTION 5: Binding Parsing (~200 lines)
// ============================================================================

impl Parser {
    /// binding: name-binding | tuple-binding
    fn parse_bind(&mut self) -> Result<Bind, ParseError> {
        // TODO: implement
        todo!()
    }

    /// tuple-binding: { id; ... }
    fn parse_tuple_bind(&mut self) -> Result<Bind, ParseError> {
        // TODO: implement
        todo!()
    }

    /// let-binding: LET | VAR binding = expr
    fn parse_let_bind(&mut self) -> Result<LetBind, ParseError> {
        // TODO: implement
        todo!()
    }

    // ... all binding rules
}

// ============================================================================
// SECTION 6: Expression Parsing (~400 lines)
// ============================================================================

impl Parser {
    /// expr: binop-expr | prim-expr | ...
    fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        // TODO: implement
        todo!()
    }

    /// binop-expr: expr binop expr (left-associative)
    fn parse_binop_expr(&mut self) -> Result<Expr, ParseError> {
        // TODO: implement
        todo!()
    }

    /// prim-expr: num-expr | id-expr | paren-expr | ...
    fn parse_prim_expr(&mut self) -> Result<Expr, ParseError> {
        // TODO: implement
        todo!()
    }

    /// app-expr: expr(args)
    fn parse_app_expr(&mut self, base: Expr) -> Result<Expr, ParseError> {
        // TODO: implement
        todo!()
    }

    // ... all expression rules
}

// ============================================================================
// SECTION 7: Control Flow Parsing (~300 lines)
// ============================================================================

impl Parser {
    /// if-expr: IF expr: block ... END
    fn parse_if_expr(&mut self) -> Result<Expr, ParseError> {
        // TODO: implement
        todo!()
    }

    /// cases-expr: CASES (type) expr: branches ... END
    fn parse_cases_expr(&mut self) -> Result<Expr, ParseError> {
        // TODO: implement
        todo!()
    }

    /// when-expr: WHEN expr: block END
    fn parse_when_expr(&mut self) -> Result<Expr, ParseError> {
        // TODO: implement
        todo!()
    }

    /// for-expr: FOR expr(bindings) ann: body END
    fn parse_for_expr(&mut self) -> Result<Expr, ParseError> {
        // TODO: implement
        todo!()
    }

    // ... all control flow rules
}

// ============================================================================
// SECTION 8: Function Parsing (~250 lines)
// ============================================================================

impl Parser {
    /// fun-expr: FUN name<typarams>(args) ann: doc body where END
    fn parse_fun_expr(&mut self) -> Result<Expr, ParseError> {
        // TODO: implement
        todo!()
    }

    /// lambda-expr: LAM(args) ann: doc body where END
    fn parse_lambda_expr(&mut self) -> Result<Expr, ParseError> {
        // TODO: implement
        todo!()
    }

    /// method-expr: METHOD(args) ann: doc body where END
    fn parse_method_expr(&mut self) -> Result<Expr, ParseError> {
        // TODO: implement
        todo!()
    }

    // Shared helpers
    fn parse_fun_header(&mut self) -> Result<(Vec<Name>, Vec<Bind>, Ann), ParseError> {
        // TODO: implement
        todo!()
    }

    fn parse_where_clause(&mut self) -> Result<Option<Expr>, ParseError> {
        // TODO: implement
        todo!()
    }

    // ... all function rules
}

// ============================================================================
// SECTION 9: Data Definition Parsing (~250 lines)
// ============================================================================

impl Parser {
    /// data-expr: DATA name<typarams>: variants sharing where END
    fn parse_data_expr(&mut self) -> Result<Expr, ParseError> {
        // TODO: implement
        todo!()
    }

    /// data-variant: name(members) | name
    fn parse_variant(&mut self) -> Result<Variant, ParseError> {
        // TODO: implement
        todo!()
    }

    /// data-with: with: members END
    fn parse_data_with(&mut self) -> Result<Vec<Member>, ParseError> {
        // TODO: implement
        todo!()
    }

    // ... all data definition rules
}

// ============================================================================
// SECTION 10: Table Operation Parsing (~300 lines)
// ============================================================================

impl Parser {
    /// table-expr: table: headers row: ... end
    fn parse_table_expr(&mut self) -> Result<Expr, ParseError> {
        // TODO: implement
        todo!()
    }

    /// table-select: select columns from table
    fn parse_table_select(&mut self) -> Result<Expr, ParseError> {
        // TODO: implement
        todo!()
    }

    // ... all 15 table operation rules
}

// ============================================================================
// SECTION 11: Check/Test Parsing (~200 lines)
// ============================================================================

impl Parser {
    /// check-expr: CHECK name: body END
    fn parse_check_expr(&mut self) -> Result<Expr, ParseError> {
        // TODO: implement
        todo!()
    }

    /// check-test: expr is expr | expr raises expr | ...
    fn parse_check_test(&mut self) -> Result<Expr, ParseError> {
        // TODO: implement
        todo!()
    }

    /// spy-stmt: spy: contents END
    fn parse_spy_stmt(&mut self) -> Result<Expr, ParseError> {
        // TODO: implement
        todo!()
    }

    // ... all check/spy rules
}

// ============================================================================
// SECTION 12: Helper Methods (~150 lines)
// ============================================================================

impl Parser {
    /// Parse comma-separated list
    fn parse_comma_list<T, F>(&mut self, parser: F) -> Result<Vec<T>, ParseError>
    where
        F: Fn(&mut Self) -> Result<T, ParseError>
    {
        let mut items = Vec::new();

        loop {
            items.push(parser(self)?);

            if !self.matches(TokenType::Comma) {
                break;
            }
            self.advance(); // consume comma
        }

        Ok(items)
    }

    /// Parse optional element
    fn parse_optional<T, F>(&mut self, parser: F) -> Result<Option<T>, ParseError>
    where
        F: Fn(&mut Self) -> Result<T, ParseError>
    {
        match parser(self) {
            Ok(value) => Ok(Some(value)),
            Err(_) => Ok(None),
        }
    }

    /// Parse list until END token
    fn parse_until_end<T, F>(&mut self, parser: F) -> Result<Vec<T>, ParseError>
    where
        F: Fn(&mut Self) -> Result<T, ParseError>
    {
        let mut items = Vec::new();

        while !self.matches(TokenType::End) && !self.is_at_end() {
            items.push(parser(self)?);
        }

        Ok(items)
    }

    // ... common parsing utilities
}
```

## Implementation Approach

### Single-File Benefits

**AST (ast.rs):**
- All types visible at once
- Easy to understand relationships
- Simple ctrl+f to find any type
- Matches reference implementation organization
- Clear section boundaries with comments

**Parser (parser.rs):**
- Complete grammar in one place
- Easy to see all parsing logic
- Natural grouping by AST section
- Straightforward to match BNF to implementation
- All helper methods accessible

### Key Parsing Strategies

1. **Operator Precedence**: Left-associative, same precedence for all binary operators
2. **Parenthesis Distinction**: PARENSPACE vs PARENNOSPACE critical for syntax
3. **Location Tracking**: Every node includes precise `Loc` with line/column/char
4. **Type Parameters**: Parse angle brackets for generic parameters
5. **Check Blocks**: Optional `where:` clauses on functions/data
6. **Blocky Syntax**: Track `:` vs `block:` with boolean flag

## Implementation Order

### Phase 1: Foundation ✅ COMPLETE (2025-10-31)
1. ✅ Update Cargo.toml with dependencies
2. ✅ Create complete `src/ast.rs` with all ~150 types (1,300 lines)
3. ✅ Test JSON serialization matches expected format
4. ✅ Create `src/error.rs` with ParseError types
5. ✅ Update `src/lib.rs` with module declarations
6. ✅ Write serialization unit tests (11 tests passing)
7. ✅ Create `src/parser.rs` skeleton with all sections and methods
8. ✅ Demonstrate working tokenizer and AST JSON output

**Status:** All foundation complete. Ready for Phase 2.
**See:** `PHASE1_COMPLETE.md` for detailed summary.

### Phase 2: Parser Core (Week 2) ✅ COMPLETE (2025-10-31)
1. ✅ Create `src/parser.rs` with Parser struct
2. ✅ Implement core navigation methods (peek, advance, expect, matches)
3. ✅ Implement primitive parsing (num, bool, str, id)
4. ✅ Implement name parsing for identifiers
5. ✅ Implement operator precedence for binop-expr (left-associative)
6. ✅ Test basic expressions (12 tests passing)

**Completed:**
- ✅ `parse_prim_expr()` dispatches to specific parsers based on token type
- ✅ `parse_num()` parses NUMBER tokens into SNum
- ✅ `parse_bool()` parses TRUE/FALSE tokens into SBool
- ✅ `parse_str()` parses STRING tokens into SStr
- ✅ `parse_id_expr()` parses NAME tokens into SId with SName
- ✅ `parse_rational()` and `parse_rough_rational()` stubs for fractions
- ✅ `parse_binop_expr()` with left-associative operator parsing
- ✅ `is_binop()` and `parse_binop()` helper methods
- ✅ All 15 binary operators supported (+, -, *, /, <, >, <=, >=, ==, =~, <>, <=>, and, or, ^)
- ✅ Comprehensive test suite (12 tests, all passing)
  - Number literals
  - String literals
  - Boolean literals (true/false)
  - Identifiers
  - Simple binary operations
  - Left-associative parsing
  - Multiple operators
  - JSON serialization

**Next Steps:**
- Move to Phase 3: More complex expressions (objects, arrays, tuples, function calls)

**Important Note on Operator Precedence:**
Pyret has **NO operator precedence hierarchy**. All binary operators have equal precedence and are strictly left-associative. This is a **fundamental property of the BNF grammar**, not a parser implementation detail:
- `binop-expr: expr (binop expr)*` - All operators treated equally
- `2 + 3 * 4` parses as `(2 + 3) * 4 = 20` (not `2 + (3 * 4) = 14`)
- Users must use explicit parentheses to control evaluation order
- Our parser correctly implements this grammar structure

See `OPERATOR_PRECEDENCE.md` for detailed explanation.

**Important Note on Parentheses & Function Application:**
Parentheses and function application must be implemented together due to Pyret's whitespace-sensitive syntax:
- `f(x)` - Function application (PARENNOSPACE - no space before `(`)
- `f (x)` - Function `f` applied to parenthesized expression (PARENSPACE - space before `(`)
- `(x + y)` - Parenthesized expression (PARENSPACE)

The tokenizer distinguishes between PARENNOSPACE and PARENSPACE, which determines parsing behavior. These features are tightly coupled and should be implemented together to handle the ambiguity correctly.

### Phase 3: Expressions (Week 3) ⏳ NEXT
1. ⏳ **Implement parenthesized expressions and function application together** (whitespace-sensitive)
   - `parse_paren_expr()` for `(expr)` when PARENSPACE/LPAREN
   - `parse_app_expr(base)` for `base(args)` when PARENNOSPACE
   - Update `parse_binop_expr()` to handle postfix operations
   - Tests for `f(x)` vs `f (x)` vs `(x + y)`
2. ⏳ Implement object expressions (`{ field: value }`)
3. ⏳ Implement array expressions (`[1, 2, 3]`)
4. ⏳ Implement tuple expressions (`{1; 2; 3}`)
5. ⏳ Implement dot access (`obj.field`)
6. ⏳ Implement bracket access (`obj[key]`)
7. ⏳ Add comprehensive expression tests

### Phase 4: Control Flow (Week 4)
1. Implement `parse_if_expr` and `parse_if_pipe_expr`
2. Implement `parse_cases_expr`
3. Implement `parse_when_expr`
4. Implement `parse_for_expr`
5. Add control flow tests

### Phase 5: Functions & Bindings (Week 5)
1. Implement `parse_fun_expr`, `parse_lambda_expr`, `parse_method_expr`
2. Implement `parse_fun_header` for function signatures
3. Implement `parse_bind`, `parse_tuple_bind`
4. Implement `parse_let_expr`, `parse_var_expr`, `parse_rec_expr`
5. Implement `parse_where_clause`
6. Add function/binding tests

### Phase 6: Data Definitions (Week 6)
1. Implement `parse_data_expr`
2. Implement `parse_variant` and variant constructors
3. Implement `parse_data_with` and `parse_data_sharing`
4. Add data definition tests

### Phase 7: Type System (Week 7)
1. Implement `parse_ann` dispatcher
2. Implement all annotation types (arrow, record, tuple, app, etc.)
3. Implement `parse_type_expr`, `parse_newtype_expr`
4. Implement `parse_type_let_expr`
5. Add type annotation tests

### Phase 8: Imports/Exports (Week 8)
1. Implement `parse_use_stmt`
2. Implement all import variants (include, import, import-fields)
3. Implement all provide variants
4. Implement include/provide specs
5. Add import/export tests

### Phase 9: Tables (Week 9)
1. Implement `parse_table_expr` and `parse_load_table_expr`
2. Implement all table operations (select, filter, order, extract, extend, update)
3. Add table operation tests

### Phase 10: Testing & Spy (Week 10)
1. Implement `parse_check_expr` and `parse_check_test`
2. Implement all check operators
3. Implement `parse_spy_stmt`
4. Implement `parse_reactor_expr`
5. Add testing feature tests

### Phase 11: Program Structure (Week 11)
1. Implement `parse_prelude` with imports/exports
2. Complete `parse_program`
3. Implement `parse_block` at top level
4. Add integration tests with full programs

### Phase 12: Polish (Week 12)
1. Improve error messages with better context
2. Add comprehensive test suite
3. Compare JSON output with `node ast-to-json.jarr`
4. Update CLAUDE.md with documentation
5. Performance optimization
6. Write usage examples

## Testing Strategy

### Unit Tests
- Each parse function tested independently
- Test valid inputs produce correct AST
- Test invalid inputs produce appropriate errors
- Test edge cases (empty lists, optional fields, etc.)

### Snapshot Tests
- Use `insta` crate for snapshot testing
- Serialize AST to JSON and compare against snapshots
- Easy to review changes with `cargo insta review`

### Reference Comparison Tests
1. Create test harness that:
   - Parses Pyret file with Rust parser
   - Parses same file with `node ast-to-json.jarr`
   - Compares JSON outputs (accounting for float precision, etc.)
2. Test against Pyret standard library files
3. Test against example programs

### Integration Tests
- Parse complete Pyret programs
- Verify all AST nodes are correctly constructed
- Test round-trip: parse → serialize → parse

## Reference Files

- **BNF Grammar:** `/Users/jimmyhmiller/Documents/Code/open-source/pyret-lang/src/js/base/pyret-grammar.bnf`
- **JS Parser:** `/Users/jimmyhmiller/Documents/Code/open-source/pyret-lang/src/js/trove/parse-pyret.js`
- **AST Definitions:** `/Users/jimmyhmiller/Documents/Code/open-source/pyret-lang/src/arr/trove/ast.arr`
- **AST to JSON:** `node ast-to-json.jarr <file.arr>` in pyret-lang directory

## Success Criteria

✅ `src/ast.rs` contains all ~150 AST types (~3500 lines)
✅ `src/parser.rs` contains all parser logic (~3000 lines)
✅ JSON output matches `node ast-to-json.jarr` exactly
✅ Comprehensive test coverage (>90%)
✅ Clear section organization in both files
✅ Documentation in CLAUDE.md
✅ Can parse all Pyret standard library files
✅ Parser handles all BNF grammar rules
✅ Error messages are clear and helpful
✅ Performance is acceptable (<10ms for typical files)

## Next Steps

1. Start with Phase 1: Create `src/ast.rs` with all AST types
2. Set up serde serialization and test JSON output
3. Create `src/error.rs` for error handling
4. Begin parser skeleton in `src/parser.rs`
5. Incrementally implement each parser section
6. Test continuously against reference implementation
