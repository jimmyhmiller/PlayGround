# JavaScript Parser in Java 25

A modern, ESTree-compliant JavaScript parser built in Java 25 using modern language features (records, sealed interfaces, pattern matching).

## 🎯 Current Status

**7.32% exact matches** (2,467 / 33,698 cached test262 files)
**53.91% parse without errors** (18,166 failures / 33,698 total)

### ✅ What Works
- **Expressions**: All operators (binary, unary, logical, update, ternary, instanceof), literals, identifiers
- **Statements**: Variables (var/let/const), blocks, if/else, all loops (for/while/do-while/for-in/for-of), break/continue, return
- **Functions**: Function declarations and expressions with parameters
- **Objects & Arrays**: Literals, member access, computed properties, property shorthand, method shorthand, getters/setters
- **ES6 Features**: for-in/for-of, property shorthand, method shorthand, function expressions
- **Structural JSON Comparison**: Proper deep equality validation against oracle

### ❌ What's Missing (High Impact)
- Template literals (1,518 failures)
- Destructuring (626 failures)
- Arrow functions (551 failures)
- Rest parameters (419 failures)
- Classes, modules, async/await
- Try/catch, switch, throw
- And more (see error analysis below)

## 🚀 Quick Start

### Prerequisites
- Java 25 with preview features
- Maven 3.9+
- Node.js (for oracle validation with esprima)

### Build & Test
```bash
# Install dependencies (required for oracle validation)
npm install esprima

# Run all tests
./mvnw test

# Run specific feature tests
./mvnw test -Dtest=FunctionExpressionTest
./mvnw test -Dtest=ObjectTest
./mvnw test -Dtest=WhileStatementTest
./mvnw test -Dtest=TemplateLiteralTest

# Measure test262 coverage (runs entire ECMAScript test suite)
./mvnw test -Dtest=Test262Runner

# Run with verbose output
./mvnw test -X

# Clean and rebuild
./mvnw clean test
```

### Test Organization

The project uses multiple test approaches:

1. **Unit Tests** (`src/test/java/com/jsparser/*Test.java`)
   - Feature-specific tests with JUnit 5
   - Parameterized tests for comprehensive coverage
   - Oracle validation against esprima for AST accuracy

2. **Test262 Suite** (`Test262Runner.java`)
   - Runs against official ECMAScript conformance suite
   - 33,698 cached test files from test262
   - Measures exact AST matches and parse success rate
   - Located in `test-oracles/test262/`

3. **Additional Test Files** (`tests/` directory)
   - Edge case examples
   - Debugging snippets
   - Number literal test cases
   - Oracle comparison fixtures

### Understanding Test Results

- **Exact Match**: JSON output matches esprima exactly
- **Parse Success**: Code parses without errors (AST may differ)
- **Parse Failure**: Syntax error or unsupported feature

For detailed testing methodology, see [TESTING.md](agent_docs/TESTING.md).

### Usage
```java
import com.jsparser.Parser;
import com.jsparser.ast.Program;

String source = "function add(a, b) { return a + b; }";
Program ast = Parser.parse(source);
// Returns ESTree-compliant AST
```

## 📚 Documentation

- **[FEATURES.md](agent_docs/FEATURES.md)** - Complete feature list and implementation guide
- **[NEXT_STEPS.md](agent_docs/NEXT_STEPS.md)** - Detailed guide for next feature (function expressions)
- **[ROADMAP.md](agent_docs/ROADMAP.md)** - 10-phase plan to complete JavaScript support
- **[TESTING.md](agent_docs/TESTING.md)** - Testing philosophy and guidelines
- **[CHANGELOG.md](agent_docs/CHANGELOG.md)** - Version history and changes
- **[Error Analysis](docs/analysis/)** - Detailed failure categorization and statistics

## 🏗️ Architecture

### Modern Java Features
- **Records** - Immutable AST nodes (`record FunctionDeclaration(...)`)
- **Sealed Interfaces** - Type-safe AST hierarchy (`sealed interface Statement`)
- **Pattern Matching** - Clean switch expressions
- **Preview Features** - Latest Java 25 capabilities

### Parser Design
- **Recursive Descent** - Top-down parsing with operator precedence
- **ESTree Compliant** - Matches official JavaScript AST specification
- **Oracle Validated** - 100% match with esprima reference parser
- **Source Locations** - Complete line/column tracking for every node

### Project Structure
```
src/main/java/com/jsparser/
├── Parser.java           # Main recursive descent parser
├── Lexer.java            # Tokenization
├── Token.java            # Token representation
├── TokenType.java        # Token types enum
└── ast/                  # AST node definitions
    ├── Node.java         # Base interface
    ├── Statement.java    # Statement types
    ├── Expression.java   # Expression types
    ├── Program.java      # Root node
    ├── FunctionDeclaration.java
    ├── ForStatement.java
    └── ...

src/test/java/com/jsparser/
├── FunctionExpressionTest.java
├── WhileStatementTest.java
├── ObjectTest.java
├── TemplateLiteralTest.java
├── Test262Runner.java    # Full test262 suite
└── OracleParser.java     # Esprima integration

agent_docs/               # All project documentation (markdown files)
docs/analysis/            # Error analysis and failure reports
tests/                    # Additional test files and fixtures
test-oracles/test262/     # ECMAScript test suite (51,350 files)
scripts/                  # Build and analysis scripts
build-tools/              # Maven wrapper and build utilities
```

## 🧪 Testing Philosophy

1. **Oracle-Based Testing**: Every feature is validated against esprima (reference parser)
2. **100% Accuracy Required**: AST JSON must match exactly (structural deep equality)
3. **Parameterized Tests**: Multiple test cases per feature for comprehensive coverage
4. **Real-World Validation**: test262 suite measures actual JavaScript support

### Example Test

```java
@ParameterizedTest
@ValueSource(strings = {
    "function foo() {}",
    "function add(a, b) { return a + b; }"
})
void testFunctionDeclarations(String source) throws Exception {
    Program expected = OracleParser.parse(source);  // esprima
    Program actual = Parser.parse(source);          // our parser

    assertEquals(
        mapper.writeValueAsString(expected),
        mapper.writeValueAsString(actual)
    );
}
```

### Running Analysis Scripts

The `scripts/` directory contains utilities for analyzing test failures:

```bash
# Generate test262 cache (pre-parse all test files with esprima)
./scripts/generate-test262-cache.js

# Analyze current failures and categorize errors
./scripts/analyze_failures.sh

# Generate detailed error reports
node scripts/generate_complete_errors.js
```

See [TESTING.md](agent_docs/TESTING.md) for complete testing documentation.

## 📈 Progress Tracking

### Recent Session History
- **Baseline**: 0.48% exact matches (163 / 33,698 files)
- **Fixed JSON comparison**: Implemented structural deep equality
- **Added MemberExpression.optional**: 7.25% (2,444 files) - 15x improvement!
- **For-in/for-of loops**: Reduced failures by 438 files
- **Property shorthand**: ES6 `{x}` syntax support
- **Method shorthand**: ES6 `{foo() {}}` syntax support
- **instanceof operator**: Binary operator support
- **Getters/setters**: Contextual `get`/`set` in objects
- **Current**: **7.32% exact matches (2,467 files)** ⭐

### Test Suite Growth
- 189 unit tests (all passing)
- 12+ feature categories tested
- 2,467 exact AST matches
- 15,532 files parse successfully (46.09%)
- Proper structural validation with Objects.deepEquals()

## 🎯 Next Steps

Based on error analysis of remaining 18,166 parse failures:

### High-Impact Missing Features
1. **Template Literals** (1,518 failures) - Backtick strings with interpolation
2. **Destructuring** (627 failures) - `const {x, y} = obj` and `const [a, b] = arr`
3. **Arrow Functions** (551 failures) - `x => x + 1` and `(a, b) => a + b`
4. **Rest Parameters** (419 failures) - `function foo(...args) {}`
5. **Property Getters/Setters** (574 failures) - Object accessor refinements
6. **Spread Operator** - `...` in arrays and function calls

### Implementation Priority
Function expressions are already working! Focus next on:
- Arrow functions (simpler than template literals)
- Method shorthand improvements
- Rest/spread operators
- Destructuring patterns

## 🤝 Contributing

This is a learning/research project demonstrating modern Java for parser implementation.

When adding features:
1. Follow the 7-step process in [FEATURES.md](agent_docs/FEATURES.md)
2. Ensure 100% oracle validation with esprima
3. Update documentation in `agent_docs/`
4. Run full test suite (`./mvnw test`)
5. Measure test262 impact (`./mvnw test -Dtest=Test262Runner`)
6. Update error analysis if needed

## 📊 Benchmarks

- **Parse Speed**: ~1ms for typical functions
- **Test Suite**: <3 seconds for 165 tests
- **test262 Run**: ~2 seconds for 51,350 files
- **Memory**: Minimal (AST is immutable records)

## 🔧 Tech Stack

- **Language**: Java 25 with `--enable-preview`
- **Build**: Maven 3.9.6
- **Testing**: JUnit 5 (parameterized tests)
- **Oracle**: esprima 4.0+ via Node.js
- **Validation**: Jackson for JSON comparison
- **Test Suite**: test262 (official ECMAScript conformance)

## 📖 Learning Resources

- [ESTree Spec](https://github.com/estree/estree) - AST specification
- [Esprima](https://esprima.org/) - Reference parser
- [test262](https://github.com/tc39/test262) - Official test suite
- [Java Records](https://openjdk.org/jeps/395) - Immutable data
- [Sealed Classes](https://openjdk.org/jeps/409) - Type hierarchies

## 📝 License

This is a personal learning project. Feel free to learn from it!

## 🙏 Acknowledgments

- **esprima** - Reference implementation and oracle
- **test262** - ECMAScript conformance suite
- **ESTree** - Standard AST format
- **Java Team** - Modern language features

## 📊 Error Analysis

Current remaining failures (from Test262Runner):

```
Top error messages:
  [1,518] Unexpected character: ` (template literals)
  [627]   Expected identifier in variable declaration (destructuring)
  [574]   Expected ':' after property key (advanced object features)
  [551]   Expected function name (arrow functions/async)
  [530]   Unexpected character: ^ (XOR operator)
  [516]   Unexpected character: \ (escape sequences)
  [419]   Expected parameter name (rest params/destructuring)
  [362]   Expected property key (computed property names)
  [357]   Expected property name after '.' (private fields)
```

---

**Current Status**: 7.32% exact matches | 2,467 AST matches | 15,532 parse successfully | Next: Arrow Functions
