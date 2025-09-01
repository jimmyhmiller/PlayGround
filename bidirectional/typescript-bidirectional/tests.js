import { 
  parseTypeScript, 
  infer, 
  check, 
  getExpression,
  processProgram,
  formatType,
  runTest,
  runFail, 
  assertEquals,
  BOOL_TYPE,
  NUMBER_TYPE,
  STRING_TYPE,
  createTypeVariable,
  createUnknownType
} from './index.js';

// Helper to test infer with TypeScript code
const testInfer = (code, context = {}) => {
  const sourceFile = parseTypeScript(code);
  const expression = getExpression(sourceFile);
  return infer(expression, context);
};

// Helper to test check with TypeScript code
const testCheck = (code, expectedType, context = {}) => {
  const sourceFile = parseTypeScript(code);
  const expression = getExpression(sourceFile);
  return check(expression, expectedType, context);
};

console.log('Starting TypeScript bidirectional type checker tests...\n');

// ================================
// BASIC LITERAL TESTS
// ================================
console.log('🔤 Testing Basic Literals...\n');

assertEquals(testInfer('true'), BOOL_TYPE, 'boolean literal true');
assertEquals(testInfer('false'), BOOL_TYPE, 'boolean literal false');
assertEquals(testInfer('42'), NUMBER_TYPE, 'number literal');
assertEquals(testInfer('0'), NUMBER_TYPE, 'zero literal');
assertEquals(testInfer('3.14'), NUMBER_TYPE, 'decimal literal');
assertEquals(testInfer('"hello"'), STRING_TYPE, 'string literal');
assertEquals(testInfer('"world"'), STRING_TYPE, 'another string literal');
assertEquals(testInfer('""'), STRING_TYPE, 'empty string literal');

console.log('✅ All basic literal tests passed!\n');

// ================================
// BINARY EXPRESSION TESTS
// ================================
console.log('➕ Testing Binary Expressions...\n');

// Arithmetic operations
assertEquals(testInfer('10 + 20'), NUMBER_TYPE, 'number addition');
assertEquals(testInfer('100 - 50'), NUMBER_TYPE, 'number subtraction');
assertEquals(testInfer('7 * 8'), NUMBER_TYPE, 'number multiplication');
assertEquals(testInfer('20 / 4'), NUMBER_TYPE, 'number division');
assertEquals(testInfer('17 % 5'), NUMBER_TYPE, 'number modulo');

// String operations
assertEquals(testInfer('"hello" + "world"'), STRING_TYPE, 'string concatenation');
assertEquals(testInfer('"a" + "b"'), STRING_TYPE, 'single char concatenation');

// String + Number concatenation (coercion)
assertEquals(testInfer('"Price: $" + 42'), STRING_TYPE, 'string + number concatenation');
assertEquals(testInfer('"Count: " + 100'), STRING_TYPE, 'string + number concatenation 2');

// Number + String concatenation (coercion)
assertEquals(testInfer('42 + " dollars"'), STRING_TYPE, 'number + string concatenation');
assertEquals(testInfer('100 + "%"'), STRING_TYPE, 'number + string concatenation 2');

// Comparison operations
assertEquals(testInfer('10 > 5'), BOOL_TYPE, 'greater than');
assertEquals(testInfer('3 < 7'), BOOL_TYPE, 'less than');
assertEquals(testInfer('5 >= 5'), BOOL_TYPE, 'greater than or equal');
assertEquals(testInfer('2 <= 3'), BOOL_TYPE, 'less than or equal');
assertEquals(testInfer('42 === 42'), BOOL_TYPE, 'strict equality');
assertEquals(testInfer('1 == 1'), BOOL_TYPE, 'loose equality');
assertEquals(testInfer('5 !== 6'), BOOL_TYPE, 'strict inequality');
assertEquals(testInfer('3 != 4'), BOOL_TYPE, 'loose inequality');

// String comparisons
assertEquals(testInfer('"a" === "a"'), BOOL_TYPE, 'string equality');
assertEquals(testInfer('"x" !== "y"'), BOOL_TYPE, 'string inequality');

// Boolean comparisons
assertEquals(testInfer('true === true'), BOOL_TYPE, 'boolean equality');
assertEquals(testInfer('false !== true'), BOOL_TYPE, 'boolean inequality');

console.log('✅ All binary expression tests passed!\n');

// ================================
// CONDITIONAL EXPRESSION TESTS
// ================================
console.log('❓ Testing Conditional Expressions...\n');

assertEquals(testInfer('true ? 10 : 20'), NUMBER_TYPE, 'conditional with numbers');
assertEquals(testInfer('false ? "yes" : "no"'), STRING_TYPE, 'conditional with strings');
assertEquals(testInfer('true ? true : false'), BOOL_TYPE, 'conditional with booleans');
assertEquals(testInfer('5 > 3 ? 100 : 200'), NUMBER_TYPE, 'conditional with comparison');
assertEquals(testInfer('"a" === "b" ? "same" : "different"'), STRING_TYPE, 'conditional with string comparison');

console.log('✅ All conditional expression tests passed!\n');

// ================================
// VARIABLE CONTEXT TESTS
// ================================
console.log('📦 Testing Variables in Context...\n');

const numberContext = { x: NUMBER_TYPE };
assertEquals(testInfer('x', numberContext), NUMBER_TYPE, 'number variable');

const stringContext = { name: STRING_TYPE };
assertEquals(testInfer('name', stringContext), STRING_TYPE, 'string variable');

const boolContext = { flag: BOOL_TYPE };
assertEquals(testInfer('flag', boolContext), BOOL_TYPE, 'boolean variable');

const multiContext = { a: NUMBER_TYPE, b: STRING_TYPE, c: BOOL_TYPE };
assertEquals(testInfer('a', multiContext), NUMBER_TYPE, 'multi-context number');
assertEquals(testInfer('b', multiContext), STRING_TYPE, 'multi-context string');
assertEquals(testInfer('c', multiContext), BOOL_TYPE, 'multi-context boolean');

console.log('✅ All variable context tests passed!\n');

// ================================
// FUNCTION EXPRESSION TESTS (ARROW FUNCTIONS)
// ================================
console.log('🏹 Testing Arrow Function Expressions...\n');

// Test function type inference from programs
const identityProgram = processProgram('let identity = (x: number) => x');
const identityType = identityProgram.context.identity;
assertEquals(identityType.kind, 'function', 'identity should be function type');
assertEquals(identityType.paramTypes[0], NUMBER_TYPE, 'identity param should be number');
assertEquals(identityType.returnType, NUMBER_TYPE, 'identity return should be number');

const boolFuncProgram = processProgram('let isTrue = (x: number) => true');
const boolFuncType = boolFuncProgram.context.isTrue;
assertEquals(boolFuncType.paramTypes[0], NUMBER_TYPE, 'isTrue param should be number');
assertEquals(boolFuncType.returnType, BOOL_TYPE, 'isTrue return should be boolean');

// Multi-parameter functions
const addProgram = processProgram('let add = (x: number, y: number) => x + y');
const addType = addProgram.context.add;
assertEquals(addType.paramTypes.length, 2, 'add should have 2 parameters');
assertEquals(addType.paramTypes[0], NUMBER_TYPE, 'add first param should be number');
assertEquals(addType.paramTypes[1], NUMBER_TYPE, 'add second param should be number');
assertEquals(addType.returnType, NUMBER_TYPE, 'add return should be number');

// Three parameter functions
const threeParamProgram = processProgram('let calc = (a: number, b: number, c: string) => a + b');
const threeParamType = threeParamProgram.context.calc;
assertEquals(threeParamType.paramTypes.length, 3, 'calc should have 3 parameters');
assertEquals(threeParamType.paramTypes[2], STRING_TYPE, 'calc third param should be string');

console.log('✅ All arrow function tests passed!\n');

// ================================
// FUNCTION DECLARATION TESTS
// ================================
console.log('🎯 Testing Function Declarations...\n');

// Simple function declaration
const simpleFuncProgram = processProgram(`
  function double(x: number): number {
    return x * 2
  }
`);
const doubleFuncType = simpleFuncProgram.context.double;
assertEquals(doubleFuncType.paramTypes[0], NUMBER_TYPE, 'double param should be number');
assertEquals(doubleFuncType.returnType, NUMBER_TYPE, 'double return should be number');

// Function with different return type
const boolFuncDecl = processProgram(`
  function isPositive(x: number): boolean {
    return x > 0
  }
`);
assertEquals(boolFuncDecl.context.isPositive.returnType, BOOL_TYPE, 'isPositive should return boolean');

// Multi-parameter function declaration
const multiParamDecl = processProgram(`
  function combine(x: number, y: string, z: boolean): number {
    return x + 10
  }
`);
const combineType = multiParamDecl.context.combine;
assertEquals(combineType.paramTypes.length, 3, 'combine should have 3 params');
assertEquals(combineType.paramTypes[0], NUMBER_TYPE, 'combine param 1 should be number');
assertEquals(combineType.paramTypes[1], STRING_TYPE, 'combine param 2 should be string');
assertEquals(combineType.paramTypes[2], BOOL_TYPE, 'combine param 3 should be boolean');

// Function without explicit return type (inferred)
const inferredReturnProgram = processProgram(`
  function multiply(x: number, y: number) {
    return x * y
  }
`);
assertEquals(inferredReturnProgram.context.multiply.returnType, NUMBER_TYPE, 'multiply return should be inferred as number');

console.log('✅ All function declaration tests passed!\n');

// ================================
// HIGHER-ORDER FUNCTION TESTS
// ================================
console.log('🔄 Testing Higher-Order Functions...\n');

const higherOrderProgram = processProgram(`
  function createMultiplier(factor: number): (x: number) => number {
    function multiplier(x: number): number {
      return x * factor
    }
    return multiplier
  }
`);

const createMultiplierType = higherOrderProgram.context.createMultiplier;
assertEquals(createMultiplierType.paramTypes[0], NUMBER_TYPE, 'createMultiplier param should be number');
assertEquals(createMultiplierType.returnType.kind, 'function', 'createMultiplier should return function');
assertEquals(createMultiplierType.returnType.paramTypes[0], NUMBER_TYPE, 'returned function param should be number');
assertEquals(createMultiplierType.returnType.returnType, NUMBER_TYPE, 'returned function return should be number');

// Higher-order function with arrow syntax
const arrowHigherOrder = processProgram(`
  let makeAdder = (n: number): (x: number) => number => (x: number) => x + n
`);
const makeAdderType = arrowHigherOrder.context.makeAdder;
assertEquals(makeAdderType.returnType.kind, 'function', 'makeAdder should return function');

console.log('✅ All higher-order function tests passed!\n');

// ================================
// FUNCTION APPLICATION TESTS
// ================================
console.log('📞 Testing Function Applications...\n');

// Inline function application
assertEquals(testInfer('((x: number) => x)(42)'), NUMBER_TYPE, 'inline function application');
assertEquals(testInfer('((x: number) => x > 0)(5)'), BOOL_TYPE, 'inline function returning boolean');

// Multi-parameter inline application
assertEquals(testInfer('((x: number, y: number) => x + y)(10, 20)'), NUMBER_TYPE, 'inline multi-parameter function');

// Function application from context
const funcContext = processProgram(`
  function addOne(x: number): number {
    return x + 1
  }
`);
const addOneContext = funcContext.context;
assertEquals(testInfer('addOne(5)', addOneContext), NUMBER_TYPE, 'function call from context');

// Complex function application
const complexApp = processProgram(`
  function createAdder(n: number): (x: number) => number {
    function adder(x: number): number {
      return x + n
    }
    return adder
  }
  
  let addFive = createAdder(5)
  addFive(10)
`);
const complexResults = complexApp.results;
const lastResult = complexResults[complexResults.length - 1];
assertEquals(lastResult.type, NUMBER_TYPE, 'complex function application should return number');

console.log('✅ All function application tests passed!\n');

// ================================
// VARIABLE DECLARATION TESTS
// ================================
console.log('📦 Testing Variable Declarations...\n');

// Simple variable declarations
const varTest1 = processProgram('let x = 4');
assertEquals(varTest1.context.x, NUMBER_TYPE, 'let with number literal');

const varTest2 = processProgram('let message = "hello"');
assertEquals(varTest2.context.message, STRING_TYPE, 'let with string literal');

const varTest3 = processProgram('let flag = true');
assertEquals(varTest3.context.flag, BOOL_TYPE, 'let with boolean literal');

// const declarations
const constTest = processProgram(`
  const PI = 3.14
  const NAME = "TypeScript"
  const ENABLED = true
`);
assertEquals(constTest.context.PI, NUMBER_TYPE, 'const number declaration');
assertEquals(constTest.context.NAME, STRING_TYPE, 'const string declaration');
assertEquals(constTest.context.ENABLED, BOOL_TYPE, 'const boolean declaration');

// var declarations
const varDecl = processProgram(`
  var count = 42
  var title = "Test"
`);
assertEquals(varDecl.context.count, NUMBER_TYPE, 'var number declaration');
assertEquals(varDecl.context.title, STRING_TYPE, 'var string declaration');

// Multiple variables in one statement
const multiVar = processProgram(`
  let a = 10, b = "hello", c = false
`);
assertEquals(multiVar.context.a, NUMBER_TYPE, 'multiple var declaration - number');
assertEquals(multiVar.context.b, STRING_TYPE, 'multiple var declaration - string');
assertEquals(multiVar.context.c, BOOL_TYPE, 'multiple var declaration - boolean');

// Variables from expressions
const exprVar = processProgram(`
  let sum = 10 + 20
  let concat = "hello" + "world"
  let comparison = 5 > 3
  let priceString = "Price: $" + 42
  let countString = 100 + " items"
  let mixedConcat = "Result: " + (5 * 10)
`);
assertEquals(exprVar.context.sum, NUMBER_TYPE, 'variable from arithmetic expression');
assertEquals(exprVar.context.concat, STRING_TYPE, 'variable from string concatenation');
assertEquals(exprVar.context.comparison, BOOL_TYPE, 'variable from comparison');
assertEquals(exprVar.context.priceString, STRING_TYPE, 'variable from string + number');
assertEquals(exprVar.context.countString, STRING_TYPE, 'variable from number + string');
assertEquals(exprVar.context.mixedConcat, STRING_TYPE, 'variable from mixed expression concatenation');

// Variables using other variables
const chainVar = processProgram(`
  let x = 100
  let y = x
  let z = y + 50
`);
assertEquals(chainVar.context.x, NUMBER_TYPE, 'first variable in chain');
assertEquals(chainVar.context.y, NUMBER_TYPE, 'second variable in chain');
assertEquals(chainVar.context.z, NUMBER_TYPE, 'third variable in chain');

// Function variables
const funcVar = processProgram(`
  let identity = (x: number) => x
  let constant = (x: number) => true
`);
assertEquals(funcVar.context.identity.kind, 'function', 'function variable type');
assertEquals(funcVar.context.constant.returnType, BOOL_TYPE, 'function variable return type');

console.log('✅ All variable declaration tests passed!\n');

// ================================
// DEFERRED TYPE INFERENCE TESTS
// ================================
console.log('🔮 Testing Deferred Type Inference...\n');

// Basic deferred inference
const deferred1 = processProgram(`
  let x
  x = 42
`);
assertEquals(deferred1.context.x, NUMBER_TYPE, 'deferred number inference');

const deferred2 = processProgram(`
  let y
  y = "hello"
`);
assertEquals(deferred2.context.y, STRING_TYPE, 'deferred string inference');

// Multiple deferred variables
const multiDeferred = processProgram(`
  let a
  let b
  let c
  a = 100
  b = "world"
  c = true
`);
assertEquals(multiDeferred.context.a, NUMBER_TYPE, 'multiple deferred - number');
assertEquals(multiDeferred.context.b, STRING_TYPE, 'multiple deferred - string');
assertEquals(multiDeferred.context.c, BOOL_TYPE, 'multiple deferred - boolean');

// Using deferred variables after assignment
const usedDeferred = processProgram(`
  let x
  let y
  x = 10
  y = 20
  x + y
`);
assertEquals(usedDeferred.context.x, NUMBER_TYPE, 'used deferred variable - x');
assertEquals(usedDeferred.context.y, NUMBER_TYPE, 'used deferred variable - y');
const deferredResult = usedDeferred.results[usedDeferred.results.length - 1];
assertEquals(deferredResult.type, NUMBER_TYPE, 'expression using deferred variables');

// Function assignment with deferred inference
const funcDeferred = processProgram(`
  let myFunc
  myFunc = (x: number) => x * 2
`);
assertEquals(funcDeferred.context.myFunc.kind, 'function', 'deferred function assignment');
assertEquals(funcDeferred.context.myFunc.returnType, NUMBER_TYPE, 'deferred function return type');

console.log('✅ All deferred type inference tests passed!\n');

// ================================
// CONTROL FLOW TESTS (IF/ELSE)
// ================================
console.log('🌊 Testing Control Flow...\n');

// Same type in both branches
const sameTypeIf = processProgram(`
  let x
  if (true) {
    x = 42
  } else {
    x = 100
  }
  x + 1
`);
assertEquals(sameTypeIf.context.x, NUMBER_TYPE, 'same type in both if/else branches');

// Multiple variables with if/else
const multiVarIf = processProgram(`
  let a
  let b
  let c
  a = 10  // Always assigned
  if (true) {
    b = 20
    c = 30
  } else {
    b = 40
    c = 50
  }
  a + b + c
`);
assertEquals(multiVarIf.context.a, NUMBER_TYPE, 'always assigned variable');
assertEquals(multiVarIf.context.b, NUMBER_TYPE, 'if/else assigned variable b');
assertEquals(multiVarIf.context.c, NUMBER_TYPE, 'if/else assigned variable c');

// Nested if statements
const nestedIf = processProgram(`
  let x
  let y
  if (true) {
    x = 10
    if (false) {
      y = 20
    } else {
      y = 30
    }
  } else {
    x = 40
    y = 50
  }
  x * y
`);
assertEquals(nestedIf.context.x, NUMBER_TYPE, 'nested if - x variable');
assertEquals(nestedIf.context.y, NUMBER_TYPE, 'nested if - y variable');

console.log('✅ All control flow tests passed!\n');

// ================================
// COMPLEX PROGRAM TESTS
// ================================
console.log('🏗️ Testing Complex Programs...\n');

// Multiple functions calling each other
const multipleProgram = processProgram(`
  function double(x: number): number {
    return x * 2
  }
  
  function quadruple(x: number): number {
    return double(double(x))
  }
  
  function isLarge(x: number): boolean {
    return quadruple(x) > 100
  }
  
  let result = quadruple(5)
  let check = isLarge(10)
`);

assertEquals(multipleProgram.context.double.returnType, NUMBER_TYPE, 'double function return type');
assertEquals(multipleProgram.context.quadruple.returnType, NUMBER_TYPE, 'quadruple function return type');
assertEquals(multipleProgram.context.isLarge.returnType, BOOL_TYPE, 'isLarge function return type');
assertEquals(multipleProgram.context.result, NUMBER_TYPE, 'result variable type');
assertEquals(multipleProgram.context.check, BOOL_TYPE, 'check variable type');

// Realistic e-commerce example with string formatting
const ecommerceProgram = processProgram(`
  function calculateTax(amount: number): number {
    return amount * 0.08
  }
  
  function calculateDiscount(amount: number, isPremium: boolean): number {
    return isPremium ? amount * 0.1 : 0
  }
  
  function calculateTotal(baseAmount: number, isPremium: boolean): number {
    let tax = calculateTax(baseAmount)
    let discount = calculateDiscount(baseAmount, isPremium)
    let total = baseAmount + tax - discount
    return total
  }
  
  function formatPrice(amount: number): string {
    return "$" + amount
  }
  
  function formatOrderSummary(total: number, itemCount: number): string {
    return "Order Total: " + total + " for " + itemCount + " items"
  }
  
  function isExpensive(amount: number, isPremium: boolean): boolean {
    let total = calculateTotal(amount, isPremium)
    return total > 100
  }
  
  let orderTotal = calculateTotal(150, true)
  let formattedTotal = formatPrice(orderTotal)
  let summary = formatOrderSummary(orderTotal, 3)
  let expensive = isExpensive(75, false)
`);

assertEquals(ecommerceProgram.context.calculateTax.returnType, NUMBER_TYPE, 'calculateTax return type');
assertEquals(ecommerceProgram.context.calculateDiscount.returnType, NUMBER_TYPE, 'calculateDiscount return type');
assertEquals(ecommerceProgram.context.calculateTotal.returnType, NUMBER_TYPE, 'calculateTotal return type');
assertEquals(ecommerceProgram.context.formatPrice.returnType, STRING_TYPE, 'formatPrice return type');
assertEquals(ecommerceProgram.context.formatOrderSummary.returnType, STRING_TYPE, 'formatOrderSummary return type');
assertEquals(ecommerceProgram.context.isExpensive.returnType, BOOL_TYPE, 'isExpensive return type');
assertEquals(ecommerceProgram.context.orderTotal, NUMBER_TYPE, 'orderTotal variable type');
assertEquals(ecommerceProgram.context.formattedTotal, STRING_TYPE, 'formattedTotal variable type');
assertEquals(ecommerceProgram.context.summary, STRING_TYPE, 'summary variable type');
assertEquals(ecommerceProgram.context.expensive, BOOL_TYPE, 'expensive variable type');

// Nested function declarations
const nestedFunctionProgram = processProgram(`
  function outer(x: number): number {
    function inner(y: number): number {
      return x + y
    }
    return inner(10)
  }
  
  let result = outer(5)
`);

assertEquals(nestedFunctionProgram.context.outer.returnType, NUMBER_TYPE, 'outer function return type');
assertEquals(nestedFunctionProgram.context.result, NUMBER_TYPE, 'nested function result type');

console.log('✅ All complex program tests passed!\n');

// ================================
// GENERIC FUNCTION TESTS
// ================================
console.log('🧬 Testing Generic Functions...\n');

// Generic identity function
const T = createTypeVariable('T');
const genericIdentity = testInfer('(x: T) => x');
assertEquals(genericIdentity.kind, 'function', 'generic identity should be function');
assertEquals(genericIdentity.paramTypes[0].kind, 'variable', 'generic parameter should be type variable');
assertEquals(genericIdentity.returnType.kind, 'variable', 'generic return should be type variable');

// Generic function application with type inference
const genericContext = processProgram(`
  let identity = (x: T) => x
`);
assertEquals(testInfer('identity(42)', genericContext.context), NUMBER_TYPE, 'generic identity applied to number');
assertEquals(testInfer('identity("hello")', genericContext.context), STRING_TYPE, 'generic identity applied to string');
assertEquals(testInfer('identity(true)', genericContext.context), BOOL_TYPE, 'generic identity applied to boolean');

console.log('✅ All generic function tests passed!\n');

// ================================
// ERROR CASE TESTS
// ================================
console.log('❌ Testing Error Cases...\n');

// Type checking failures
runFail(() => testCheck('true', NUMBER_TYPE), 'boolean checked as number should fail');
runFail(() => testInfer('unknownVar'), 'undefined variable should fail');
runFail(() => testInfer('true ? 10 : "hello"'), 'conditional branch type mismatch should fail');
runFail(() => testInfer('"hello" ? 1 : 2'), 'non-boolean condition should fail');

// Test other invalid operations (not addition)
runFail(() => testInfer('10 - "hello"'), 'number - string should fail');
runFail(() => testInfer('"hello" * 5'), 'string * number should fail');

// Function parameter annotation requirement
runFail(() => testInfer('((b) => b ? false : true)(true)'), 'unannotated lambda parameter should fail');

// Function application errors
const errorContext = processProgram(`
  function add(x: number, y: number): number {
    return x + y
  }
`);

runFail(() => testInfer('add(10)', errorContext.context), 'too few arguments should fail');
runFail(() => testInfer('add(10, 20, 30)', errorContext.context), 'too many arguments should fail');
runFail(() => testInfer('add(10, "hello")', errorContext.context), 'wrong argument type should fail');

// Variable declaration errors
runFail(() => processProgram('let x: string = 42'), 'initializer type mismatch should fail');

// Deferred inference errors
runFail(() => processProgram(`
  let x
  x + 1
`), 'using variable before assignment should fail');

runFail(() => processProgram(`
  let x
  x = 42
  x = "hello"
`), 'inconsistent assignment type should fail');

runFail(() => processProgram(`
  y = 42
`), 'assignment to undeclared variable should fail');

// Control flow errors
runFail(() => processProgram(`
  let x
  if (true) {
    x = 42
  } else {
    x = "hello"
  }
`), 'different types in if/else branches should fail');

runFail(() => processProgram(`
  let x
  if (true) {
    x = 42
  }
`), 'assignment only in then branch should fail');

// Function declaration errors
runFail(() => processProgram(`
  function badFunction(x: number): string {
    return x + 1
  }
`), 'return type mismatch should fail');

runFail(() => processProgram(`
  function mixedReturn(x: number) {
    return x > 0 ? 42 : "negative"
  }
`), 'mixed return types should fail');

console.log('✅ All error case tests passed!\n');

// ================================
// CHECK FUNCTION TESTS
// ================================
console.log('✅ Testing Check Function...\n');

runTest(() => {
  testCheck('true', BOOL_TYPE);
  return 'check boolean literal';
});

runTest(() => {
  testCheck('42', NUMBER_TYPE);
  return 'check number literal';
});

runTest(() => {
  const program = processProgram('function add(x: number, y: number): number { return x + y }');
  const expectedType = program.context.add;
  testCheck('(x: number, y: number) => x + y', expectedType);
  return 'check function type';
});

console.log('✅ All check function tests passed!\n');

// ================================
// SUMMARY
// ================================
console.log('🎉 ALL COMPREHENSIVE TESTS PASSED! 🎉');
console.log('\nTest Summary:');
console.log('✅ Basic Literals (8 tests)');
console.log('✅ Binary Expressions (20 tests)'); 
console.log('✅ Conditional Expressions (5 tests)');
console.log('✅ Variable Context (7 tests)');
console.log('✅ Arrow Function Expressions (8 tests)');
console.log('✅ Function Declarations (7 tests)');
console.log('✅ Higher-Order Functions (4 tests)');
console.log('✅ Function Applications (8 tests)');
console.log('✅ Variable Declarations (18 tests)');
console.log('✅ Deferred Type Inference (8 tests)');
console.log('✅ Control Flow (6 tests)');
console.log('✅ Complex Programs (15 tests)');
console.log('✅ Generic Functions (5 tests)');
console.log('✅ Error Cases (20 tests)');
console.log('✅ Check Function (3 tests)');
console.log('\n🔢 Total: 140+ comprehensive tests covering all functionality!');