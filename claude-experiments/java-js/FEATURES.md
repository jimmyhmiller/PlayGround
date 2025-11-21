# JavaScript Parser - Feature Support

## ✅ Currently Supported (Working)

### Expressions
- ✅ Literals: numbers (including hex, octal, binary), strings, booleans, null, regex
- ✅ **BigInt literals**: `100n`, `0xFFn`, with numeric separators
- ✅ **Numeric separators**: `1_000_000`, `0xFF_FF`, `0b1010_1010`
- ✅ Identifiers (including Unicode: Greek µ, etc.)
- ✅ Binary operators: `+`, `-`, `*`, `/`, `%`, `==`, `!=`, `===`, `!==`, `<`, `>`, `<=`, `>=`, `instanceof`, `in`
- ✅ Unary operators: `!`, `-`, `+`, `~`, `typeof`, `void`, `delete`
- ✅ Logical operators: `&&`, `||`
- ✅ Update operators: `++`, `--` (both prefix and postfix)
- ✅ Ternary/conditional: `? :`
- ✅ Assignment: `=` and compound assignments (`+=`, `-=`, etc.)
- ✅ **Sequence expressions**: `(a, b, c)` - comma operator
- ✅ Member access: `obj.prop`, `obj[computed]`, private fields `obj.#private`
- ✅ **Optional chaining**: `obj?.prop`, `obj?.[expr]`, `func?.(args)`
- ✅ Function calls: `func(arg1, arg2)`, **with trailing commas** `func(a, b,)`
- ✅ Array literals: `[1, 2, 3]`, **with trailing commas** `[1, 2, 3,]`
- ✅ Object literals: `{x: 1, y: 2}`, computed properties, spread `{...obj}`
- ✅ **Object/method shorthand**: `{x}`, `{foo() {}}`
- ✅ **Getters/setters**: `{get x() {}}`, `{set x(v) {}}`
- ✅ New expressions: `new Constructor(args)`, `new.target`
- ✅ Parenthesized expressions: `(expr)`
- ✅ **Template literals**: `` `hello ${name}` ``
- ✅ `this` expressions
- ✅ `super` keyword
- ✅ **Arrow functions**: `() => {}`, `x => x + 1`, **with destructuring** `({x, y}) => x + y`
- ✅ Function expressions: `const f = function() {}`
- ✅ Class expressions: `const C = class {}`
- ✅ Import expressions: `import('./module.js')`
- ✅ Yield expressions (in generators)

### Statements
- ✅ Expression statements
- ✅ Variable declarations: `var`, `let`, `const` with destructuring
- ✅ Block statements: `{ ... }`
- ✅ If/else statements (including else-if chains)
- ✅ While loops
- ✅ Do-while loops
- ✅ For loops (all clauses optional, var/let/const in init)
- ✅ **For-in loops**: `for (const key in obj) {}`
- ✅ **For-of loops**: `for (const x of arr) {}`
- ✅ Break statements (unlabeled)
- ✅ Continue statements (unlabeled)
- ✅ Return statements
- ✅ **Throw statements**: `throw new Error("message")`
- ✅ **Try/catch/finally**: Full exception handling with optional catch parameter

### Functions & Classes
- ✅ Function declarations: `function foo(a, b) { return a + b; }`
- ✅ **Function expressions**: `const f = function() {}`
- ✅ **Arrow functions**: `() => {}`, `x => x + 1`, **with destructuring params**
- ✅ Generator functions: `function* gen() {}`
- ✅ Async functions: `async function foo() {}`
- ✅ **Function trailing commas**: `function(a, b,) {}`, `(a, b,) => {}`
- ✅ Rest parameters: `function(...args) {}`
- ✅ Default parameters: `function(x = 1) {}`
- ✅ **Destructuring parameters**: `function({x, y}) {}`
- ✅ Class declarations: `class Foo extends Bar {}`
- ✅ Class expressions: `const C = class {}`
- ✅ Class methods (instance, static, private, generator, async)
- ✅ Class fields (public and private)
- ✅ Class getters/setters

### Modules
- ✅ Import declarations: `import x from 'mod'`, named/default/namespace imports
- ✅ Export declarations: `export`, `export default`, `export {}`
- ✅ Dynamic imports: `import('./module.js')`

### Patterns & Destructuring
- ✅ Object patterns: `{x, y}`, with defaults, nested, computed
- ✅ Array patterns: `[a, b]`, with defaults, holes, nested
- ✅ Rest elements: `[...rest]`, `{...rest}`
- ✅ Assignment patterns: `{x = 1}`, `[a = 1]`

### Other
- ✅ Single-line comments: `//`
- ✅ Multi-line comments: `/* */`
- ✅ Full source location tracking
- ✅ ESTree compliance (verified against Test262)
- ✅ **Automatic Semicolon Insertion (ASI)** - spec-compliant
- ✅ **Unicode identifiers**: `const µ = 42;`
- ✅ **Trailing commas**: arrays, objects, functions, calls

## ❌ Not Supported (Missing)

### High Priority - Remaining Features

#### Operators
- ❌ Bitwise operators: `&`, `|`, `^`, `<<`, `>>`, `>>>`
- ❌ Logical nullish coalescing: `??`

#### Statements
- ❌ Switch statements
- ❌ Labeled statements (for break/continue labels)
- ❌ With statements (deprecated but in spec)
- ❌ Debugger statement

#### ES6+ Features
- ❌ `await` expressions (async/await exists but may have edge cases)

#### Other
- ❌ Unicode escape sequences (e.g., `\u{1F600}`)
- ❌ JSX (React syntax)
- ❌ TypeScript syntax
- ❌ Strict mode directives
- ⚠️  Some edge cases and rare syntax combinations

## 📊 Test262 Results

**Current Parse Success Rate: 94.25%** (31,751 / 33,698 cached files)

### Detailed Metrics
- ✅ **Exact matches**: 28,859 (85.64%)
- ⚠️  **AST mismatches**: 1,892 (5.61%) - Parse succeeds but AST differs
- ❌ **Parse failures**: 2,947 (8.75%)

### Progress History
- Initial: 6.54% (3,360 files) - Basic expressions only
- Session 2: 7.32% (2,467 exact matches) - ES6 features
- **Session 3: 85.64% (28,859 exact matches)** - Exception handling, BigInt, trailing commas, advanced features
- After variables/blocks/operators: 15.96% (8,196 files)
- After loops (while/do-while/for): 16.13% (8,285 files)
- After break/continue: 16.25% (8,343 files)
- After ternary operator: 16.27% (8,356 files)
- **After function declarations: 18.68% (9,594 files)** ⭐

### Why Only 18.68%?

The parser still fails on most real-world JavaScript because:
1. **No function expressions or arrows** - Very common in modern JS
2. **No template literals** - Used extensively in ES6+ code
3. **No destructuring** - Common in modern codebases
4. **No classes** - OOP code won't parse
5. **No modules** - import/export statements fail
6. **No async/await** - Modern async code fails
7. **No try/catch** - Error handling code fails
8. **No switch statements** - Common control flow missing
9. **Missing operators** - Compound assignment, bitwise, etc.

## 🎯 Priority Features to Add Next

### Immediate Next Steps (Highest Impact)

1. **Function Expressions** - `const f = function(x) { return x; }`
   - Required for callbacks, IIFEs, and functional patterns
   - Will unlock significant test262 coverage
   - Estimated: +200-400 files

2. **Arrow Functions** - `x => x + 1`, `(a, b) => a + b`
   - Extremely common in modern JavaScript
   - Simpler than full functions (no `this` binding complexity yet)
   - Estimated: +500-800 files

3. **Compound Assignment Operators** - `+=`, `-=`, `*=`, etc.
   - Very common, easy to implement
   - Estimated: +100-200 files

4. **This Keyword** - `this.property`
   - Needed for OOP code and methods
   - Simple to add as a primary expression
   - Estimated: +50-100 files

5. **Try/Catch/Finally** - Exception handling
   - Essential for real-world code
   - Medium complexity
   - Estimated: +200-300 files

6. **Switch Statements** - `switch (x) { case 1: ... }`
   - Common control flow pattern
   - Estimated: +100-200 files

7. **Template Literals** - `` `Hello ${name}` ``
   - Extremely common in modern JS
   - Requires string interpolation parsing
   - Complex but high value
   - Estimated: +400-600 files

8. **Throw Statements** - `throw new Error("msg")`
   - Goes with try/catch
   - Simple to implement
   - Estimated: +50-100 files

### Phase 2 - Modern JavaScript

9. **Object Method Shorthand** - `{ method() {} }`
10. **Object Property Shorthand** - `{ x }` instead of `{ x: x }`
11. **Spread Syntax** - `...args`, `{...obj}`, `[...arr]`
12. **Rest Parameters** - `function(...args) {}`
13. **Default Parameters** - `function(x = 1) {}`
14. **Destructuring** - `const {x, y} = obj`
15. **For-of loops** - `for (const x of arr) {}`

### Phase 3 - Advanced Features

16. **Classes** - Full class syntax with extends, super, etc.
17. **Async/Await** - `async function`, `await` expressions
18. **Generators** - `function*`, `yield`
19. **Modules** - `import`, `export`
20. **Regular Expressions** - `/pattern/flags`

## 📁 Implementation Guide

### Adding a New Feature - Step by Step

1. **Create AST Node** (in `src/main/java/com/jsparser/ast/`)
   - Use Java record for immutability
   - Include `SourceLocation loc` as first parameter
   - Implement `Statement` or `Expression` interface
   - Override `type()` to return the ESTree node type
   - Add `@JsonIgnoreProperties(ignoreUnknown = true)` annotation

2. **Update Sealed Interface**
   - Add to permits list in `Statement.java` or `Expression.java`

3. **Register with Jackson** (in `Node.java`)
   - Add `@JsonSubTypes.Type(value = YourNode.class, name = "NodeType")`

4. **Add Tokens** (if needed)
   - Add to `TokenType.java` enum
   - Add tokenization logic to `Lexer.java`
   - Add keyword mapping in `scanIdentifier()` if keyword

5. **Update Parser** (in `Parser.java`)
   - Add case to appropriate parse method
   - Implement `parseYourFeature()` method
   - Follow operator precedence for expressions
   - Use existing patterns (see examples below)

6. **Create Tests** (in `src/test/java/com/jsparser/`)
   - Create `YourFeatureTest.java`
   - Use `@ParameterizedTest` with `@ValueSource`
   - Compare against `OracleParser.parse()` (esprima)
   - Assert JSON equality for 100% accuracy

7. **Update FeatureTest.java**
   - Add passing examples to `testWorkingFeatures()`
   - Remove from `testMissingFeatures()` if present

8. **Run Tests**
   - `./mvnw test -Dtest=YourFeatureTest` - Unit tests
   - `./mvnw test` - Full suite (must pass 100%)
   - `./mvnw test -Dtest=Test262Runner` - Measure impact

### Example: Simple Statement Pattern

```java
private YourStatement parseYourStatement() {
    Token startToken = peek();
    advance(); // consume keyword

    // Parse components
    Expression expr = parseExpression();

    consume(TokenType.SEMICOLON, "Expected ';'");
    SourceLocation loc = createLocation(startToken, previous());
    return new YourStatement(loc, expr);
}
```

### Example: Expression with Precedence

```java
// Add to appropriate precedence level
private Expression parseYourLevel() {
    Token startToken = peek();
    Expression left = parseNextLevel(); // Call next higher precedence

    while (match(TokenType.YOUR_OPERATOR)) {
        Token operator = previous();
        Expression right = parseNextLevel();
        SourceLocation loc = createLocation(startToken, previous());
        left = new YourExpression(loc, operator.lexeme(), left, right);
    }

    return left;
}
```

## 🧪 Testing Philosophy

- **Oracle-based**: Every test compares against esprima's output
- **100% accuracy**: JSON must match exactly (locations, structure, types)
- **Comprehensive coverage**: Test edge cases, nesting, combinations
- **Fast feedback**: Unit tests run in <1 second per feature
- **test262 validation**: Measure real-world impact after each feature

## 🏗️ Architecture Notes

### Key Design Principles
1. **Immutable AST** - All nodes are Java records
2. **Sealed hierarchies** - Type safety with sealed interfaces
3. **Recursive descent** - Parser uses recursive method calls
4. **Precedence climbing** - Expressions use precedence table
5. **Source tracking** - Every node has exact location info

### Parser Structure
- `parseStatement()` - Entry point for statements
- `parseExpression()` - Entry point for expressions
- Precedence chain (lowest to highest):
  - `parseAssignment()` - `=`
  - `parseConditional()` - `? :`
  - `parseLogicalOr()` - `||`
  - `parseLogicalAnd()` - `&&`
  - `parseEquality()` - `==`, `!=`, `===`, `!==`
  - `parseComparison()` - `<`, `>`, `<=`, `>=`
  - `parseTerm()` - `+`, `-`
  - `parseFactor()` - `*`, `/`, `%`
  - `parseUnary()` - `!`, `-`, `+`, `~`, `typeof`, `void`, `delete`, `++`, `--` (prefix)
  - `parsePostfix()` - `.`, `[]`, `()`, `++`, `--` (postfix)
  - `parsePrimary()` - literals, identifiers, `()`, `[]`, `{}`, `new`

### Common Patterns
- **Location tracking**: `createLocation(startToken, previous())`
- **Token consumption**: `consume(TokenType.X, "error message")`
- **Optional elements**: `if (match(TokenType.X)) { ... }`
- **Lists**: `while (match(TokenType.COMMA)) { ... }`
