# Next Steps: Function Expressions

## 🎯 Goal
Implement function expressions to enable callbacks, IIFEs, and functional programming patterns.

**Estimated Impact:** +200-400 test262 files
**Difficulty:** Medium (similar to function declarations)
**Time Estimate:** 1-2 hours

## 📋 What Are Function Expressions?

Function expressions are functions used as values:

```javascript
// Named function expression
const factorial = function fact(n) {
    return n <= 1 ? 1 : n * fact(n - 1);
};

// Anonymous function expression
const add = function(a, b) {
    return a + b;
};

// As callback
array.map(function(x) { return x * 2; });

// IIFE (Immediately Invoked Function Expression)
(function() {
    console.log("Hello!");
})();
```

## 🔍 Oracle Analysis

Let's check what esprima produces:

```bash
# Check simple function expression
echo 'var f = function() {};' > /tmp/test.js
node src/test/resources/oracle-parser.js /tmp/test.js
```

Expected AST structure:
- `FunctionExpression` node (implements `Expression`, not `Statement`)
- Has `id` field (can be null for anonymous functions)
- Has `params` list
- Has `body` (BlockStatement)
- Has `generator`, `expression`, `async` flags (all false for basic functions)

## 📝 Implementation Steps

### Step 1: Create FunctionExpression AST Node

**File:** `src/main/java/com/jsparser/ast/FunctionExpression.java`

```java
package com.jsparser.ast;

import com.fasterxml.jackson.annotation.JsonIgnoreProperties;
import java.util.List;

@JsonIgnoreProperties(ignoreUnknown = true)
public record FunctionExpression(
    SourceLocation loc,
    Identifier id,           // Can be null for anonymous functions
    List<Identifier> params,
    BlockStatement body,
    boolean generator,
    boolean expression,
    boolean async
) implements Expression {
    @Override
    public String type() {
        return "FunctionExpression";
    }
}
```

### Step 2: Update Expression Interface

**File:** `src/main/java/com/jsparser/ast/Expression.java`

Add `FunctionExpression` to the permits list:

```java
public sealed interface Expression extends Node permits
    Identifier,
    Literal,
    BinaryExpression,
    AssignmentExpression,
    MemberExpression,
    CallExpression,
    ArrayExpression,
    ObjectExpression,
    NewExpression,
    UnaryExpression,
    LogicalExpression,
    UpdateExpression,
    ConditionalExpression,
    FunctionExpression {  // <-- Add this
}
```

### Step 3: Register with Jackson

**File:** `src/main/java/com/jsparser/ast/Node.java`

Add to @JsonSubTypes:

```java
@JsonSubTypes.Type(value = FunctionExpression.class, name = "FunctionExpression"),
```

### Step 4: Update Parser - Add to parsePrimary()

**File:** `src/main/java/com/jsparser/Parser.java`

In `parsePrimary()`, add a case for FUNCTION token:

```java
case FUNCTION -> {
    Token startToken = token;
    advance(); // consume 'function'

    // Optional function name (can be null for anonymous)
    Identifier id = null;
    if (check(TokenType.IDENTIFIER)) {
        Token nameToken = peek();
        advance();
        id = new Identifier(createLocation(nameToken, nameToken), nameToken.lexeme());
    }

    // Parse parameters
    consume(TokenType.LPAREN, "Expected '(' after function");
    List<Identifier> params = new ArrayList<>();

    if (!check(TokenType.RPAREN)) {
        do {
            Token paramToken = peek();
            if (!check(TokenType.IDENTIFIER)) {
                throw new RuntimeException("Expected parameter name");
            }
            advance();
            params.add(new Identifier(createLocation(paramToken, paramToken), paramToken.lexeme()));
        } while (match(TokenType.COMMA));
    }

    consume(TokenType.RPAREN, "Expected ')' after parameters");

    // Parse body
    BlockStatement body = parseBlockStatement();

    SourceLocation loc = createLocation(startToken, previous());
    yield new FunctionExpression(loc, id, params, body, false, false, false);
}
```

### Step 5: Create Tests

**File:** `src/test/java/com/jsparser/FunctionExpressionTest.java`

```java
package com.jsparser;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.jsparser.ast.Program;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;
import static org.junit.jupiter.api.Assertions.*;

public class FunctionExpressionTest {
    private static final ObjectMapper mapper = new ObjectMapper();

    @ParameterizedTest
    @ValueSource(strings = {
        "var f = function() {};",
        "var add = function(a, b) { return a + b; };",
        "var fact = function factorial(n) { return n <= 1 ? 1 : n * factorial(n - 1); };",
        "(function() {})();",
        "var x = function(a, b, c) { var sum = a + b + c; return sum; };",
        "array.map(function(x) { return x * 2; });",
        "setTimeout(function() { x++; }, 1000);",
        "var obj = { method: function(x) { return x; } };",
        "[1, 2].forEach(function(n) { console.log(n); });",
        "var f = function() { if (true) return 1; };",
        "var nested = function() { return function() { return 42; }; };",
    })
    void testFunctionExpressionsAgainstOracle(String source) throws Exception {
        Program expected = OracleParser.parse(source);
        Program actual = Parser.parse(source);

        String expectedJson = mapper.writeValueAsString(expected);
        String actualJson = mapper.writeValueAsString(actual);

        System.out.println("Testing: " + source);
        if (!expectedJson.equals(actualJson)) {
            System.out.println("EXPECTED:");
            System.out.println(mapper.writerWithDefaultPrettyPrinter().writeValueAsString(expected));
            System.out.println("\nACTUAL:");
            System.out.println(mapper.writerWithDefaultPrettyPrinter().writeValueAsString(actual));
        }

        assertEquals(expectedJson, actualJson, "AST mismatch for: " + source);
    }
}
```

### Step 6: Update FeatureTest

**File:** `src/test/java/com/jsparser/FeatureTest.java`

Add to working features:
```java
assertDoesNotThrow(() -> Parser.parse("var f = function() {};"));
assertDoesNotThrow(() -> Parser.parse("var add = function(a, b) { return a + b; };"));
```

### Step 7: Run Tests

```bash
# Unit tests for function expressions
./mvnw test -Dtest=FunctionExpressionTest

# Full test suite (all must pass)
./mvnw test

# Measure test262 impact
./mvnw test -Dtest=Test262Runner
```

## 🐛 Common Issues and Solutions

### Issue 1: "function" in Expression Context
**Problem:** Parser might not recognize `function` in expression position
**Solution:** Add FUNCTION case to `parsePrimary()`, not just `parseStatement()`

### Issue 2: IIFE Parentheses
**Problem:** `(function(){})()` might fail
**Solution:** This should work automatically - parentheses are handled in `parsePrimary()`, then the call is handled in `parsePostfix()`

### Issue 3: Named vs Anonymous
**Problem:** Mixing up when `id` should be null
**Solution:** Use `check(TokenType.IDENTIFIER)` after `function` keyword - if present, parse it; otherwise id = null

### Issue 4: Nested Functions
**Problem:** Function expressions inside function bodies
**Solution:** Should work automatically since we recursively parse statements

## ✅ Validation Checklist

- [ ] FunctionExpression.java created with correct fields
- [ ] Expression.java permits list updated
- [ ] Node.java @JsonSubTypes updated
- [ ] Parser.parsePrimary() handles FUNCTION token
- [ ] Optional function name (id can be null) handled
- [ ] Parameters parsed correctly (including empty list)
- [ ] Body parsed as BlockStatement
- [ ] All 11 test cases pass
- [ ] Full test suite passes (165+ tests)
- [ ] test262 coverage improved (+200-400 files expected)
- [ ] FeatureTest.java updated

## 📊 Expected Results

After implementing function expressions:
- Test262: ~19-20% (9,800-10,000 files)
- New passing tests: FunctionExpressionTest (11 tests)
- Total test suite: 176+ tests

## 🚀 After This - Next Features

Once function expressions work, the next highest-impact features are:

1. **Arrow Functions** - `x => x + 1`
   - Even more common in modern JS
   - Simpler syntax, no `this` binding complexity
   - Estimated: +500-800 files

2. **This Keyword** - `this.property`
   - Required for object methods
   - Very simple to add
   - Estimated: +50-100 files

3. **Compound Assignment** - `+=`, `-=`, etc.
   - Common and easy
   - Estimated: +100-200 files

## 📚 Reference Materials

- ESTree Spec: https://github.com/estree/estree/blob/master/es5.md#functionexpression
- Esprima Demo: https://esprima.org/demo/parse.html
- MDN Functions: https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Operators/function

## 💡 Tips

1. **Test incrementally**: Start with `var f = function() {};` and get that working first
2. **Use the oracle**: When stuck, compare your AST to esprima's output
3. **Copy patterns**: Look at FunctionDeclaration for guidance - FunctionExpression is very similar
4. **Check locations**: Make sure all SourceLocation objects are correct
5. **Run often**: `./mvnw test -Dtest=FunctionExpressionTest` after each change

Good luck! 🎉
