# Implementation Notes

Technical details and lessons learned from implementing the JavaScript parser.

## Table of Contents
- [Structural JSON Comparison](#structural-json-comparison)
- [For-In/For-Of Loops](#for-infor-of-loops)
- [Property and Method Shorthand](#property-and-method-shorthand)
- [Contextual Keywords](#contextual-keywords)
- [instanceof Operator](#instanceof-operator)
- [Common Pitfalls](#common-pitfalls)

---

## Structural JSON Comparison

### Problem
Initially used string comparison of JSON output:
```java
String expectedJson = mapper.writeValueAsString(expected);
String actualJson = mapper.writeValueAsString(actual);
assertEquals(expectedJson, actualJson);  // FRAGILE!
```

This failed when field ordering differed, even though ASTs were structurally identical.

### Solution
Parse JSON back into objects and use deep equality:
```java
Object expectedObj = mapper.readValue(expectedJson, Object.class);
Object actualObj = mapper.readValue(actualJson, Object.class);
assertTrue(Objects.deepEquals(expectedObj, actualObj));  // ROBUST!
```

### Impact
- **163 → 2,444 exact matches** (15x improvement!)
- Handles field ordering differences
- Properly compares nested structures
- Jackson's Map and List implement equals() correctly

---

## For-In/For-Of Loops

### Challenge
JavaScript has three types of for loops that look similar syntactically:
```javascript
for (var i = 0; i < 10; i++) {}     // Regular for
for (var x in obj) {}                // for-in
for (var x of arr) {}                // for-of
```

### Approach
Unified parsing strategy:
1. Parse the initializer (variable declaration or expression)
2. Check for `in` or `of` keyword
3. Branch to appropriate statement type

```java
private Statement parseForStatement() {
    // ... parse init ...

    if (check(TokenType.IN)) {
        advance();
        Expression right = parseExpression();
        return new ForInStatement(left, right, body);
    } else if (check(TokenType.OF)) {
        advance();
        Expression right = parseExpression();
        return new ForOfStatement(left, right, body);
    }

    // Otherwise regular for loop
    consume(TokenType.SEMICOLON, ...);
    // ... continue with test and update ...
}
```

### Key Insights
- `in` and `of` are keywords in for-loop context
- Need to add them to lexer's keyword list
- `left` can be either VariableDeclaration or Expression (for destructuring)
- Must not consume semicolon before checking for in/of

---

## Property and Method Shorthand

### ES6 Syntax Forms
Objects support multiple property syntaxes:
```javascript
{
  x: 1,              // Regular property
  x,                 // Shorthand (ES6) - equivalent to x: x
  foo() {},          // Method shorthand (ES6)
  get x() {},        // Getter
  set x(v) {},       // Setter
  [expr]: value      // Computed property name
}
```

### Implementation Strategy

#### Property Shorthand
After parsing identifier key, check for colon:
```java
if (check(TokenType.COLON)) {
    // Regular property: {x: value}
    consume(TokenType.COLON);
    value = parseAssignment();
} else if (!computed && key instanceof Identifier &&
           (check(TokenType.COMMA) || check(TokenType.RBRACE))) {
    // Shorthand: {x} means {x: x}
    value = key;
    shorthand = true;
}
```

#### Method Shorthand
After identifier, check for opening paren:
```java
if (check(TokenType.LPAREN) && !computed) {
    // Method: {foo() {}}
    isMethod = true;
    consume(TokenType.LPAREN);
    List<Identifier> params = parseParameters();
    consume(TokenType.RPAREN);
    BlockStatement body = parseBlockStatement();
    value = new FunctionExpression(null, params, body, ...);
}
```

### Pitfalls
- Must check `!computed` to avoid treating `{[x]()}` incorrectly
- Shorthand only valid for identifiers, not string/number keys
- Method creates anonymous FunctionExpression (id is null)

---

## Contextual Keywords

### Problem
`get` and `set` are keywords in object literals but valid identifiers elsewhere:
```javascript
var get = 1;           // Valid - get is identifier
{get x() {}}           // Invalid - get is keyword here
{get: 1}               // Valid - get is property name
```

### Solution
Don't add to lexer keywords - handle contextually in parser:
```java
// In object parsing only:
if (check(TokenType.IDENTIFIER)) {
    String ident = peek().lexeme();
    if (ident.equals("get") || ident.equals("set")) {
        Token nextToken = tokens.get(current + 1);
        // Only keyword if followed by property key
        if (isPropertyKey(nextToken.type())) {
            advance();
            kind = ident;
            keyToken = peek();
        }
    }
}
```

### Why This Matters
- Adding GET/SET to lexer keywords broke 2,000+ files
- Variables named `get` or `set` became parse errors
- Lookahead check prevents this issue
- Context-sensitive parsing is essential for JavaScript

---

## instanceof Operator

### Precedence Level
`instanceof` is a binary operator at comparison level (same as `<`, `>`, etc.):
```javascript
x instanceof Array < 5  // Parses as: (x instanceof Array) < 5
```

### Implementation
Add to comparison parsing:
```java
private Expression parseComparison() {
    Expression left = parseTerm();

    while (match(TokenType.LT, TokenType.LE, TokenType.GT, TokenType.GE,
                 TokenType.INSTANCEOF, TokenType.IN)) {
        Token operator = previous();
        Expression right = parseTerm();
        left = new BinaryExpression(operator.lexeme(), left, right);
    }

    return left;
}
```

### Note on `in` operator
Added `in` to comparison level as well:
```javascript
'x' in obj        // Property existence check
0 in arr          // Array index check
```

This is valid JavaScript and works at same precedence as `instanceof`.

---

## Common Pitfalls

### 1. Token Position Tracking
**Problem**: After calling `advance()`, `previous()` returns the consumed token, but `peek()` returns the next token. Easy to mix up when calculating positions.

**Solution**: Save start token before advancing:
```java
Token startToken = peek();  // Save BEFORE advance
advance();
// ... later ...
SourceLocation loc = createLocation(startToken, previous());
```

### 2. Optional Fields in AST
**Problem**: Adding new field to existing AST node breaks all construction sites.

**Solution**: Use default parameter constructors:
```java
public record MemberExpression(
    SourceLocation loc,
    Expression object,
    Expression property,
    boolean computed,
    boolean optional  // NEW FIELD
) implements Expression {
    // Backwards compatibility constructor
    public MemberExpression(SourceLocation loc, Expression object,
                           Expression property, boolean computed) {
        this(loc, object, property, computed, false);  // Default false
    }
}
```

### 3. Jackson Serialization Order
**Problem**: Jackson may serialize fields in different order than reference parser.

**Solution**: Use `Objects.deepEquals()` on parsed JSON objects, not string comparison.

### 4. Sealed Interface Permits List
**Problem**: Adding new AST node type requires updating permits clause:
```java
// This will fail to compile:
public record ForInStatement(...) implements Statement {}

// Must update Statement interface:
public sealed interface Statement extends Node permits
    ...,
    ForInStatement,  // ADD THIS
    ForOfStatement   // AND THIS
    ...
```

**Solution**: Remember to update sealed interface when adding new subtypes.

### 5. @JsonSubTypes Registration
**Problem**: Jackson needs to know about all AST node types for deserialization.

**Solution**: Update `@JsonSubTypes` annotation in Node.java:
```java
@JsonSubTypes({
    ...,
    @JsonSubTypes.Type(value = ForInStatement.class, name = "ForInStatement"),
    @JsonSubTypes.Type(value = ForOfStatement.class, name = "ForOfStatement"),
    ...
})
```

---

## Performance Considerations

### Parser Speed
- Current: ~2.5 seconds to process 33,698 test262 files
- ~0.074ms per file average
- Bottleneck is file I/O, not parsing logic

### Memory Usage
- Immutable AST (records) enables efficient garbage collection
- No need for object pooling
- Parser instances are lightweight (just token list and position)

### Optimization Opportunities
1. **Intern string tokens** - Reduce memory for common keywords
2. **Lazy source locations** - Only compute when needed
3. **Parallel test262 runs** - Process files concurrently
4. **Token lookahead buffer** - Reduce array access for peek()

---

## Testing Strategy

### Oracle-Based Validation
Every feature validated against esprima:
```java
@ParameterizedTest
@ValueSource(strings = {
    "for (var x in obj) {}",
    "for (let x of arr) {}"
})
void testForLoops(String source) throws Exception {
    Program expected = OracleParser.parse(source);  // esprima
    Program actual = Parser.parse(source);          // ours

    Object expectedObj = mapper.readValue(
        mapper.writeValueAsString(expected), Object.class);
    Object actualObj = mapper.readValue(
        mapper.writeValueAsString(actual), Object.class);

    assertTrue(Objects.deepEquals(expectedObj, actualObj));
}
```

### Test262 Runner
Automated testing against entire ECMAScript test suite:
- Skips files with unsupported features (modules, async, classes)
- Caches esprima output to avoid repeated Node.js calls
- Reports: exact matches, AST mismatches, parse failures
- Tracks error messages to guide next feature implementation

---

## Lessons Learned

1. **Start with structural comparison** - Don't waste time debugging field ordering issues
2. **Contextual keywords matter** - JavaScript has many context-sensitive tokens
3. **Lookahead is your friend** - Sometimes you need to peek ahead 2+ tokens
4. **Test incrementally** - Add feature, validate against oracle immediately
5. **Let test262 guide you** - Error analysis reveals highest-impact missing features
6. **Sealed interfaces are strict** - Every new type requires updating permits list
7. **Records are perfect for AST** - Immutability and conciseness are ideal
8. **Oracle validation is gold** - 100% confidence that implementation is correct

---

## Future Improvements

### Parser Architecture
- [ ] Add error recovery (continue parsing after errors)
- [ ] Better error messages with context
- [ ] Support for comments in AST
- [ ] Directive tracking (e.g., "use strict")

### Performance
- [ ] Parallel test262 execution
- [ ] String interning for tokens
- [ ] Lazy source location computation
- [ ] Token lookahead buffer

### Features
- [ ] Arrow functions (high priority)
- [ ] Template literals (complex, high impact)
- [ ] Destructuring patterns
- [ ] Rest/spread operators
- [ ] Classes
- [ ] Modules

### Tooling
- [ ] AST pretty-printer
- [ ] Source map generation
- [ ] REPL for testing
- [ ] Benchmark suite
