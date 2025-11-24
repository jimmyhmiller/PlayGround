# JavaScript Parser Implementation Roadmap

**Current Status: 6.54% of test262 passing (3,360 / 51,350 files)**

This document outlines the feature-by-feature implementation plan to achieve full ES2024 JavaScript parsing.

---

## Phase 1: Foundation Statements (Target: ~20% test262)

### 1.1 Variable Declarations ⭐⭐⭐ (HIGHEST PRIORITY)
**Effort:** Medium | **Impact:** Very High | **Blocks:** Almost everything

**Features:**
- `var x;` - variable declaration without initializer
- `var x = 1;` - variable declaration with initializer
- `var x = 1, y = 2;` - multiple declarators
- `let x = 1;` - block-scoped variable
- `const x = 1;` - constant declaration

**AST Nodes:**
- `VariableDeclaration` (statement)
- `VariableDeclarator` (pattern + init)

**Why First:** Required by 70%+ of JavaScript files. Blocks almost all real programs.

**Test Cases:**
```javascript
var x;
var y = 42;
let a = 1, b = 2;
const PI = 3.14;
```

---

### 1.2 Block Statements ⭐⭐⭐
**Effort:** Easy | **Impact:** High | **Blocks:** All control flow

**Features:**
- `{ statements }` - block statement
- Nested blocks
- Empty blocks `{}`

**AST Nodes:**
- `BlockStatement`

**Why Next:** Required for if/for/while/functions. Very simple to implement.

**Test Cases:**
```javascript
{ }
{ x = 1; }
{ let x = 1; { let y = 2; } }
```

---

### 1.3 Return Statements ⭐⭐
**Effort:** Easy | **Impact:** Medium | **Blocks:** Functions

**Features:**
- `return;` - return without value
- `return x;` - return with value
- `return x + y;` - return with expression

**AST Nodes:**
- `ReturnStatement`

**Test Cases:**
```javascript
return;
return 42;
return x + y;
```

---

## Phase 2: Essential Operators (Target: ~30% test262)

### 2.1 Unary Operators ⭐⭐⭐
**Effort:** Medium | **Impact:** High | **Blocks:** Many expressions

**Features:**
- `!x` - logical NOT
- `-x` - unary minus
- `+x` - unary plus
- `~x` - bitwise NOT
- `typeof x` - type checking
- `void x` - void operator
- `delete obj.prop` - property deletion

**AST Nodes:**
- `UnaryExpression`

**Operator Precedence:** Higher than binary, lower than postfix

**Test Cases:**
```javascript
!true;
-42;
+x;
typeof x;
void 0;
delete obj.prop;
```

---

### 2.2 Logical Operators ⭐⭐⭐
**Effort:** Easy | **Impact:** High | **Blocks:** Conditionals

**Features:**
- `x && y` - logical AND
- `x || y` - logical OR
- `x ?? y` - nullish coalescing (ES2020)

**AST Nodes:**
- `LogicalExpression`

**Operator Precedence:** Lower than comparison, higher than ternary

**Test Cases:**
```javascript
x && y;
a || b;
x ?? y;
a && b || c;
```

---

### 2.3 Update Operators ⭐⭐
**Effort:** Easy | **Impact:** Medium

**Features:**
- `x++` - postfix increment
- `++x` - prefix increment
- `x--` - postfix decrement
- `--x` - prefix decrement

**AST Nodes:**
- `UpdateExpression` (prefix: bool, operator: "++" | "--")

**Test Cases:**
```javascript
x++;
++x;
x--;
--x;
```

---

### 2.4 Conditional (Ternary) Operator ⭐⭐
**Effort:** Easy | **Impact:** Medium

**Features:**
- `x ? y : z` - conditional expression

**AST Nodes:**
- `ConditionalExpression`

**Operator Precedence:** Above assignment, below logical OR

**Test Cases:**
```javascript
x ? y : z;
a ? b : c ? d : e;
```

---

## Phase 3: Control Flow (Target: ~45% test262)

### 3.1 If Statements ⭐⭐⭐
**Effort:** Medium | **Impact:** High

**Features:**
- `if (test) consequent` - simple if
- `if (test) consequent else alternate` - if-else
- Nested if statements

**AST Nodes:**
- `IfStatement`

**Test Cases:**
```javascript
if (x) y;
if (x) { y; }
if (x) y; else z;
if (x) { y; } else { z; }
if (x) y; else if (z) w;
```

---

### 3.2 While/Do-While Loops ⭐⭐
**Effort:** Easy | **Impact:** Medium

**Features:**
- `while (test) body` - while loop
- `do body while (test)` - do-while loop

**AST Nodes:**
- `WhileStatement`
- `DoWhileStatement`

**Test Cases:**
```javascript
while (x) { }
do { } while (x);
```

---

### 3.3 For Loops ⭐⭐⭐
**Effort:** Hard | **Impact:** High

**Features:**
- `for (init; test; update) body` - traditional for
- `for (var x = 0; x < 10; x++)` - with variable declaration
- `for (;;)` - infinite loop

**AST Nodes:**
- `ForStatement`

**Test Cases:**
```javascript
for (;;) { }
for (let i = 0; i < 10; i++) { }
for (x = 0; x < 10; x++) { }
```

---

### 3.4 Break/Continue Statements ⭐
**Effort:** Easy | **Impact:** Low

**Features:**
- `break;` - break statement
- `continue;` - continue statement
- `break label;` - labeled break
- `continue label;` - labeled continue

**AST Nodes:**
- `BreakStatement`
- `ContinueStatement`

**Test Cases:**
```javascript
break;
continue;
break label;
```

---

## Phase 4: Functions (Target: ~60% test262)

### 4.1 Function Declarations ⭐⭐⭐
**Effort:** Hard | **Impact:** Very High

**Features:**
- `function name() {}` - basic function
- `function name(params) {}` - with parameters
- `function name(a, b, c) {}` - multiple parameters

**AST Nodes:**
- `FunctionDeclaration`
- `FunctionExpression`

**Test Cases:**
```javascript
function foo() { }
function bar(x) { return x; }
function baz(a, b) { return a + b; }
```

---

### 4.2 Function Expressions ⭐⭐⭐
**Effort:** Medium | **Impact:** High

**Features:**
- `var f = function() {}` - anonymous function expression
- `var f = function name() {}` - named function expression

**AST Nodes:**
- `FunctionExpression` (reuse from 4.1)

**Test Cases:**
```javascript
var f = function() { };
var g = function name() { };
(function() { })();
```

---

### 4.3 Arrow Functions ⭐⭐⭐
**Effort:** Hard | **Impact:** Very High | **ES Version:** ES2015

**Features:**
- `() => expr` - no params, expression body
- `x => expr` - single param, no parens
- `(x, y) => expr` - multiple params
- `(x) => { statements }` - block body

**AST Nodes:**
- `ArrowFunctionExpression`

**Test Cases:**
```javascript
() => 42;
x => x + 1;
(x, y) => x + y;
x => { return x; };
```

---

### 4.4 This Expression ⭐
**Effort:** Trivial | **Impact:** Low

**Features:**
- `this` - this keyword

**AST Nodes:**
- `ThisExpression`

**Test Cases:**
```javascript
this;
this.x;
```

---

## Phase 5: Advanced Statements (Target: ~70% test262)

### 5.1 Try/Catch/Finally ⭐⭐
**Effort:** Medium | **Impact:** Medium

**Features:**
- `try { } catch (e) { }` - try-catch
- `try { } finally { }` - try-finally
- `try { } catch (e) { } finally { }` - all three
- `try { } catch { }` - catch without binding (ES2019)

**AST Nodes:**
- `TryStatement`
- `CatchClause`

**Test Cases:**
```javascript
try { } catch (e) { }
try { } finally { }
try { } catch (e) { } finally { }
try { } catch { }
```

---

### 5.2 Throw Statement ⭐⭐
**Effort:** Easy | **Impact:** Medium

**Features:**
- `throw expr;` - throw expression

**AST Nodes:**
- `ThrowStatement`

**Test Cases:**
```javascript
throw new Error("test");
throw x;
```

---

### 5.3 Switch Statements ⭐
**Effort:** Medium | **Impact:** Low

**Features:**
- `switch (discriminant) { cases }` - switch statement
- `case value:` - case clause
- `default:` - default clause

**AST Nodes:**
- `SwitchStatement`
- `SwitchCase`

**Test Cases:**
```javascript
switch (x) {
  case 1: break;
  case 2: break;
  default: break;
}
```

---

### 5.4 Labeled Statements ⭐
**Effort:** Easy | **Impact:** Low

**Features:**
- `label: statement` - labeled statement

**AST Nodes:**
- `LabeledStatement`

**Test Cases:**
```javascript
label: x = 1;
outer: for (;;) { }
```

---

## Phase 6: ES2015+ Expressions (Target: ~80% test262)

### 6.1 Template Literals ⭐⭐⭐
**Effort:** Hard | **Impact:** Very High | **ES Version:** ES2015

**Features:**
- `` `string` `` - template literal
- `` `hello ${x}` `` - with substitution
- `` `a ${x} b ${y} c` `` - multiple substitutions
- `tag`hello`` - tagged templates

**AST Nodes:**
- `TemplateLiteral`
- `TemplateElement`
- `TaggedTemplateExpression`

**Lexer Changes:** Major - need to track template state

**Test Cases:**
```javascript
`hello`;
`hello ${world}`;
`${a} ${b}`;
tag`hello ${x}`;
```

---

### 6.2 Spread Operator ⭐⭐
**Effort:** Medium | **Impact:** High | **ES Version:** ES2015

**Features:**
- `[...arr]` - array spread
- `func(...args)` - call spread
- `{...obj}` - object spread (ES2018)

**AST Nodes:**
- `SpreadElement`

**Test Cases:**
```javascript
[...arr];
func(...args);
[1, ...arr, 2];
{...obj};
```

---

### 6.3 Destructuring ⭐⭐
**Effort:** Very Hard | **Impact:** High | **ES Version:** ES2015

**Features:**
- `var [a, b] = arr;` - array destructuring
- `var {x, y} = obj;` - object destructuring
- `var {x: a} = obj;` - renamed properties
- `var [a, ...rest] = arr;` - rest in destructuring
- `var {x = 1} = obj;` - defaults

**AST Nodes:**
- `ArrayPattern`
- `ObjectPattern`
- `AssignmentPattern`
- `RestElement`

**Test Cases:**
```javascript
var [a, b] = arr;
var {x, y} = obj;
var {x: a, y: b} = obj;
var [a, ...rest] = arr;
var {x = 1} = obj;
```

---

### 6.4 For-Of Loop ⭐⭐
**Effort:** Medium | **Impact:** Medium | **ES Version:** ES2015

**Features:**
- `for (var x of arr) { }` - for-of loop
- `for (const x of arr) { }` - with const

**AST Nodes:**
- `ForOfStatement`

**Test Cases:**
```javascript
for (var x of arr) { }
for (const x of arr) { }
```

---

### 6.5 For-In Loop ⭐
**Effort:** Easy | **Impact:** Low

**Features:**
- `for (var x in obj) { }` - for-in loop

**AST Nodes:**
- `ForInStatement`

**Test Cases:**
```javascript
for (var x in obj) { }
```

---

## Phase 7: Classes & Advanced Functions (Target: ~85% test262)

### 7.1 Class Declarations ⭐⭐⭐
**Effort:** Very Hard | **Impact:** High | **ES Version:** ES2015

**Features:**
- `class Name { }` - basic class
- `class Name extends Base { }` - inheritance
- `constructor() { }` - constructor method
- `method() { }` - methods
- `static method() { }` - static methods
- `get prop() { }` - getters
- `set prop(v) { }` - setters

**AST Nodes:**
- `ClassDeclaration`
- `ClassExpression`
- `ClassBody`
- `MethodDefinition`

**Test Cases:**
```javascript
class Foo { }
class Bar extends Foo { }
class Baz {
  constructor() { }
  method() { }
  static staticMethod() { }
  get x() { }
  set x(v) { }
}
```

---

### 7.2 Super Keyword ⭐
**Effort:** Easy | **Impact:** Medium | **ES Version:** ES2015

**Features:**
- `super.method()` - super property
- `super()` - super call

**AST Nodes:**
- `Super`

**Test Cases:**
```javascript
super();
super.method();
```

---

### 7.3 Async/Await ⭐⭐
**Effort:** Hard | **Impact:** High | **ES Version:** ES2017

**Features:**
- `async function f() { }` - async function
- `await expr` - await expression

**AST Nodes:**
- Update `FunctionDeclaration` with `async` flag
- `AwaitExpression`

**Test Cases:**
```javascript
async function f() { }
async () => { };
await promise;
```

---

### 7.4 Generators ⭐
**Effort:** Medium | **Impact:** Low | **ES Version:** ES2015

**Features:**
- `function* gen() { }` - generator function
- `yield expr` - yield expression
- `yield* expr` - yield delegation

**AST Nodes:**
- Update `FunctionDeclaration` with `generator` flag
- `YieldExpression`

**Test Cases:**
```javascript
function* gen() { yield 1; }
yield x;
yield* otherGen();
```

---

## Phase 8: Modules (Target: ~90% test262)

### 8.1 Import Declarations ⭐⭐⭐
**Effort:** Hard | **Impact:** High | **ES Version:** ES2015

**Features:**
- `import x from 'mod';` - default import
- `import {x, y} from 'mod';` - named imports
- `import * as ns from 'mod';` - namespace import
- `import 'mod';` - side-effect import

**AST Nodes:**
- `ImportDeclaration`
- `ImportSpecifier`
- `ImportDefaultSpecifier`
- `ImportNamespaceSpecifier`

**Parser Changes:** Need to handle module vs script mode

**Test Cases:**
```javascript
import x from 'mod';
import {x, y} from 'mod';
import * as ns from 'mod';
import 'mod';
```

---

### 8.2 Export Declarations ⭐⭐⭐
**Effort:** Hard | **Impact:** High | **ES Version:** ES2015

**Features:**
- `export var x = 1;` - export declaration
- `export {x, y};` - export specifiers
- `export default x;` - default export
- `export * from 'mod';` - re-export all

**AST Nodes:**
- `ExportNamedDeclaration`
- `ExportDefaultDeclaration`
- `ExportAllDeclaration`
- `ExportSpecifier`

**Test Cases:**
```javascript
export var x = 1;
export {x, y};
export default x;
export * from 'mod';
```

---

## Phase 9: Advanced Operators & Syntax (Target: ~95% test262)

### 9.1 Compound Assignment ⭐
**Effort:** Easy | **Impact:** Low

**Features:**
- `x += y`, `x -= y`, `x *= y`, `x /= y`, `x %=y`
- `x &= y`, `x |= y`, `x ^= y`
- `x <<= y`, `x >>= y`, `x >>>= y`
- `x &&= y`, `x ||= y`, `x ??= y` (ES2021)

**Update:** `AssignmentExpression` to handle all operators

**Test Cases:**
```javascript
x += 1;
x *= 2;
x &&= y;
```

---

### 9.2 Bitwise Operators ⭐
**Effort:** Easy | **Impact:** Low

**Features:**
- `x & y` - bitwise AND
- `x | y` - bitwise OR
- `x ^ y` - bitwise XOR
- `x << y` - left shift
- `x >> y` - sign-propagating right shift
- `x >>> y` - zero-fill right shift

**Update:** `BinaryExpression` to handle bitwise operators

**Test Cases:**
```javascript
x & y;
x | y;
x ^ y;
x << 2;
x >> 2;
x >>> 2;
```

---

### 9.3 Sequence Expression ⭐
**Effort:** Easy | **Impact:** Low

**Features:**
- `x, y` - comma operator

**AST Nodes:**
- `SequenceExpression`

**Test Cases:**
```javascript
x, y;
(x, y, z);
```

---

### 9.4 Regular Expression Literals ⭐
**Effort:** Medium | **Impact:** Medium

**Features:**
- `/pattern/flags` - regex literal

**AST Nodes:**
- Update `Literal` to support regex

**Lexer Changes:** Need regex state machine

**Test Cases:**
```javascript
/abc/;
/[a-z]/gi;
/\d+/;
```

---

### 9.5 Object/Array Shorthand (ES2015) ⭐
**Effort:** Easy | **Impact:** Low

**Features:**
- `{x}` as shorthand for `{x: x}`
- `{method() {}}` - method shorthand

**Update:** `ObjectExpression` Property to set shorthand flag

**Test Cases:**
```javascript
{x, y};
{method() { }};
```

---

## Phase 10: Edge Cases & Modern Features (Target: ~100% test262)

### 10.1 Rest Parameters ⭐
**Effort:** Easy | **Impact:** Low | **ES Version:** ES2015

**Features:**
- `function f(...args) { }` - rest parameters

**AST Nodes:**
- `RestElement` in function parameters

**Test Cases:**
```javascript
function f(...args) { }
(...args) => { };
```

---

### 10.2 Default Parameters ⭐
**Effort:** Medium | **Impact:** Low | **ES Version:** ES2015

**Features:**
- `function f(x = 1) { }` - default parameters

**AST Nodes:**
- `AssignmentPattern` in parameters

**Test Cases:**
```javascript
function f(x = 1) { }
(x = 1) => { };
```

---

### 10.3 With Statement ⭐
**Effort:** Easy | **Impact:** Very Low

**Features:**
- `with (obj) { }` - with statement (deprecated)

**AST Nodes:**
- `WithStatement`

**Test Cases:**
```javascript
with (obj) { }
```

---

### 10.4 Debugger Statement ⭐
**Effort:** Trivial | **Impact:** Very Low

**Features:**
- `debugger;` - debugger statement

**AST Nodes:**
- `DebuggerStatement`

**Test Cases:**
```javascript
debugger;
```

---

### 10.5 Empty Statement ⭐
**Effort:** Trivial | **Impact:** Very Low

**Features:**
- `;` - empty statement

**AST Nodes:**
- `EmptyStatement`

**Test Cases:**
```javascript
;
;;
```

---

### 10.6 Computed Member in New ⭐
**Effort:** Easy | **Impact:** Very Low

**Features:**
- Fix edge cases in `new` expression with computed members

**Test Cases:**
```javascript
new obj[key]();
new obj.prop[key]();
```

---

## Implementation Strategy

### Test-Driven Approach
For each feature:

1. **Create Test File** - `src/test/java/com/jsparser/Feature_X_Test.java`
   - Test against oracle (esprima)
   - Include edge cases
   - Test combinations with existing features

2. **Add AST Nodes** - `src/main/java/com/jsparser/ast/`
   - Create record classes
   - Add to sealed interface
   - Register with Jackson

3. **Update Lexer** (if needed) - `src/main/java/com/jsparser/Lexer.java`
   - Add new tokens
   - Update `TokenType` enum

4. **Update Parser** - `src/main/java/com/jsparser/Parser.java`
   - Add parsing methods
   - Handle operator precedence
   - Update existing methods if needed

5. **Verify Oracle Match** - Run tests
   - All tests must pass
   - JSON output must match esprima exactly

6. **Run test262** - Check progress
   - Should see incremental improvement
   - Document new pass rate

---

## Priority Legend

⭐⭐⭐ = Critical (blocks many features, high test262 impact)
⭐⭐ = Important (medium test262 impact)
⭐ = Nice to have (low test262 impact)

---

## Estimated Timeline

- **Phase 1-2:** 2-3 weeks → ~30% test262
- **Phase 3-4:** 3-4 weeks → ~60% test262
- **Phase 5-6:** 4-5 weeks → ~80% test262
- **Phase 7-8:** 4-5 weeks → ~90% test262
- **Phase 9-10:** 2-3 weeks → ~100% test262

**Total:** ~15-20 weeks for full ES2024 support

---

## Success Metrics

- ✅ All features pass oracle tests (100% esprima match)
- ✅ Complete source location tracking
- ✅ Proper operator precedence
- ✅ test262 pass rate > 95%
- ✅ Modern Java features used (records, pattern matching, sealed interfaces)

---

## Next Step

**START HERE:** Implement Variable Declarations (Phase 1.1)

This single feature will unlock ~10-15% more test262 files and is required by almost everything else.
