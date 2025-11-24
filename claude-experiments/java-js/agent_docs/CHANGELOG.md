# Changelog

All notable changes to this JavaScript parser project.

## [Unreleased]

### Session 3 - Exception Handling & Advanced Features (2025-11-18)

#### Added
- **Throw statements** - `throw new Error("message")` syntax
  - Created `ThrowStatement` AST node
  - Enforces no line terminator between `throw` and expression per ECMAScript spec
  - Integrated with ASI (Automatic Semicolon Insertion)
- **Try/catch/finally statements** - Full exception handling support
  - Created `TryStatement` and `CatchClause` AST nodes
  - Support for optional catch parameter (ES2019+): `catch { }`
  - Support for catch with parameter: `catch (e) { }`
  - Optional finally block
  - Validates at least one of catch or finally is present
- **Array trailing commas** - ES5+ syntax `[1, 2, 3,]`
  - Parser checks for closing bracket after comma
  - Properly ignores trailing comma in element list
- **Numeric separators** - ES2021 syntax for readability
  - Support for underscores in all number formats: `1_000_000`
  - Hexadecimal: `0xFF_FF`
  - Octal: `0o77_77`
  - Binary: `0b1010_1010`
  - Decimal with exponent: `1e10_00`
  - Lexer strips underscores before parsing
- **BigInt literals** - ES2020 arbitrary precision integers
  - Support for `n` suffix: `100n`, `0xFFn`, `0o77n`, `0b1010n`
  - Works with numeric separators: `1_000n`
  - Stored as string representation (Java lacks native BigInt)
- **Function trailing commas** - ES2017 syntax
  - Function declarations: `function foo(a, b,) {}`
  - Function expressions: `function(a, b,) {}`
  - Arrow functions: `(a, b,) => {}`
  - Method shorthand: `{foo(a, b,) {}}`
  - Call expressions: `foo(1, 2,)`
  - Implemented across all parameter/argument parsing contexts
- **Arrow functions with destructuring parameters** - ES6 advanced syntax
  - Object destructuring: `({x, y}) => x + y`
  - Array destructuring: `([a, b]) => a + b`
  - With defaults: `({x = 1, y = 2}) => x + y`
  - New lookahead algorithm scans for matching `)` and `=>`
  - Uses depth tracking for nested brackets/braces/parens
  - Parameters parsed as patterns (supports full destructuring)
- **Sequence expressions** - Comma operator support
  - `(a, b, c)` evaluates all expressions, returns last
  - Created `SequenceExpression` AST node
  - Properly integrates with arrow function detection
  - Each element parsed as assignment expression

#### Changed
- **ASI (Automatic Semicolon Insertion)** improvements
  - Enhanced with statement-starting keyword detection
  - Handles `import`, `export`, `function`, `class`, `const`, `let`, `var`
  - More spec-compliant semicolon insertion
- **Unicode identifier support** enhanced
  - Updated `isAlpha()` to use `Character.isUnicodeIdentifierStart()`
  - Updated `isAlphaNumeric()` to use `Character.isUnicodeIdentifierPart()`
  - Explicit inclusion of `$` and `_` characters
  - Supports Greek µ and other Unicode identifier characters
- **Template literal whitespace handling** fixed
  - Moved `skipWhitespace()` to beginning of tokenize loop
  - Fixes templates with whitespace before `}`: `` `${ expr }` ``
- **Computed property methods** in classes
  - Removed `!computed` restriction in method parsing
  - Allows `{[Symbol.iterator]() {}}`
- **Dynamic import statements** distinguished from import declarations
  - Added lookahead check in `parseStatement()`
  - Detects `import(` pattern for dynamic imports
- **Object destructuring with defaults** in assignments
  - Support for `{x = defaultValue}` patterns
  - Creates `AssignmentPattern` nodes in object properties

#### Fixed
- **Property and SpreadElement registration** in Jackson @JsonSubTypes
  - Fixed "Could not resolve type id 'Property'" serialization errors
  - Added missing AST node registrations
- **Template literal edge cases** with interpolations
  - Fixed whitespace handling in `${}` expressions
- **Lexer number parsing** for special formats
  - Fixed hex, octal, binary parsing with separators and BigInt

#### Performance
- Test262 exact matches: **21,207 → 28,859** (+7,652 files, +36.1%)
- Parse success rate: **62.93% → 85.64%** (+22.71 percentage points)
- Parse failures: **31.09% → 8.75%** (-22.34 percentage points, -7,564 failures)
- Major milestones:
  - **Throw statements**: 64.95% → 76.31% (+11.36%, +3,828 files)
  - **Try/catch/finally**: 76.31% → 80.71% (+4.40%, +1,482 files)
  - **Array trailing commas**: 80.71% → 83.74% (+3.03%, +1,022 files)
  - **Function trailing commas**: 83.74% → 84.29% (+0.55%, +183 files)
  - **Arrow destructuring**: 84.29% → 85.64% (+1.35%, +456 files)
  - **Sequence expressions**: Parse failures reduced by 335 files

#### Test Coverage
- Unit tests: All passing
- test262 validation: 33,698 cached files tested
- AST mismatches: 1,892 files (5.61%)
- Parse failures: 2,947 files (8.75%)

---

### Session 2 - ES6 Features & Structural Validation (2025-11-17)

#### Added
- **Structural JSON comparison** - Replaced string comparison with `Objects.deepEquals()` for proper AST validation
- **MemberExpression.optional field** - Full support for optional chaining in AST structure
- **for-in loops** - `for (var x in obj) {}` syntax support
  - Added `IN` token type to lexer
  - Created `ForInStatement` AST node
  - Updated parser to detect and handle for-in syntax
- **for-of loops** - `for (var x of arr) {}` syntax support
  - Added `OF` token type to lexer
  - Created `ForOfStatement` AST node
  - Updated parser to detect and handle for-of syntax
- **Property shorthand** - ES6 `{x}` equivalent to `{x: x}`
  - Parser detects identifier followed by comma or close brace
  - Sets `shorthand: true` in Property node
- **Method shorthand** - ES6 `{foo() {}}` syntax
  - Parser detects identifier followed by parentheses in object context
  - Creates FunctionExpression as property value
  - Sets `method: true` in Property node
- **instanceof operator** - Binary operator support
  - Added `INSTANCEOF` token type
  - Integrated into comparison precedence level
- **Getters and setters** - `{get x() {}}` and `{set x(v) {}}` syntax
  - Contextual keyword handling for `get` and `set`
  - Lookahead to distinguish from property names
  - Sets `kind: "get"` or `kind: "set"` in Property node

#### Changed
- Parser now properly handles unified for/for-in/for-of statements
- Object property parsing supports multiple syntax forms
- Improved error messages for object property parsing

#### Performance
- Test262 exact matches: **163 → 2,467** (15x improvement, +1,400%)
- Parse success rate: **42.81% → 46.09%** (+3.28%)
- Total failures: **19,272 → 18,166** (-1,106 failures)

#### Test Coverage
- Unit tests: 165 → 189 tests (all passing)
- test262 validation: 33,698 cached files tested
- Oracle validation: 100% structural match verification

---

## Session 1 - Foundation (Previous)

### Added
- Basic expressions (binary, unary, logical, update, ternary)
- Variable declarations (var/let/const)
- Control flow (if/else, while, do-while, for)
- Break and continue statements
- Function declarations
- Objects and arrays
- Member access and computed properties
- Oracle-based testing with esprima
- ESTree-compliant AST structure

### Infrastructure
- Recursive descent parser
- Token-based lexer
- Java 25 records for AST nodes
- Sealed interfaces for type safety
- JUnit 5 parameterized tests
- test262 integration
- JSON AST comparison

---

## Key Metrics Over Time

| Session | Exact Matches | Parse Success | Parse Failures | Features Added |
|---------|--------------|---------------|----------------|----------------|
| Start   | 163 (0.48%)  | ~42%          | ~58%           | Basic parser   |
| Session 2 | 2,467 (7.32%) | 46.09%      | ~54%           | ES6 features   |
| Session 3 | 28,859 (85.64%) | 94.25%    | 8.75%          | Exception handling, BigInt, trailing commas, advanced features |

## Feature Implementation Order

1. ✅ Basic expressions and literals
2. ✅ Variable declarations
3. ✅ Control flow statements
4. ✅ Loops (for/while/do-while)
5. ✅ Break/continue
6. ✅ Function declarations
7. ✅ Objects and arrays
8. ✅ JSON structural comparison
9. ✅ for-in/for-of loops
10. ✅ Property and method shorthand
11. ✅ instanceof operator
12. ✅ Getters and setters
13. ✅ Arrow functions with destructuring
14. ✅ Throw/try/catch/finally
15. ✅ Trailing commas (arrays, functions, calls)
16. ✅ Numeric separators and BigInt
17. ✅ Sequence expressions (comma operator)
18. ✅ Unicode identifiers
19. 🔄 Template literals (basic support exists)
20. 🔄 Unicode escape sequences
21. 🔄 Remaining edge cases

## Breaking Changes

### Session 2
- None - All existing tests continue to pass
- JSON comparison now structural instead of string-based (more correct)

## Migration Guide

No breaking changes in Session 2. If you were relying on string-based JSON comparison for testing, note that the new `Objects.deepEquals()` approach is more robust and handles field ordering correctly.

## Contributors

- Parser implementation and ES6 features
- Test infrastructure and oracle validation
- Documentation and feature tracking
