import { 
  parseTypeScript, 
  synthesize, 
  check, 
  getExpression,
  processProgram,
  runTest,
  runFail, 
  assertEquals,
  BOOL_TYPE,
  NUMBER_TYPE,
  STRING_TYPE,
  createFunctionType,
  createTypeVariable
} from './index.js';

// Helper to test synthesize with TypeScript code
const testSynthesize = (code, context = {}) => {
  const sourceFile = parseTypeScript(code);
  const expression = getExpression(sourceFile);
  return synthesize(expression, context);
};

// Helper to test check with TypeScript code
const testCheck = (code, expectedType, context = {}) => {
  const sourceFile = parseTypeScript(code);
  const expression = getExpression(sourceFile);
  return check(expression, expectedType, context);
};

console.log('Starting TypeScript bidirectional type checker tests...\n');

// Basic literal tests
assertEquals(testSynthesize('true'), BOOL_TYPE, 'boolean literal');
assertEquals(testSynthesize('false'), BOOL_TYPE, 'boolean literal false');
assertEquals(testSynthesize('42'), NUMBER_TYPE, 'number literal');
assertEquals(testSynthesize('"hello"'), STRING_TYPE, 'string literal');

// Variable tests
const contextWithVar = { x: BOOL_TYPE };
assertEquals(testSynthesize('x', contextWithVar), BOOL_TYPE, 'boolean variable');

// Function tests - with type annotations
assertEquals(
  testSynthesize('(x: number) => x'),
  createFunctionType([NUMBER_TYPE], NUMBER_TYPE),
  'identity function with annotation'
);

assertEquals(
  testSynthesize('(x: number) => true'),
  createFunctionType([NUMBER_TYPE], BOOL_TYPE),
  'function number -> boolean'
);

// Multi-parameter function tests
assertEquals(
  testSynthesize('(x: number, y: string) => x + 42'),
  createFunctionType([NUMBER_TYPE, STRING_TYPE], NUMBER_TYPE),
  'two-parameter function'
);

assertEquals(
  testSynthesize('(a: number, b: number, c: string) => a + b'),
  createFunctionType([NUMBER_TYPE, NUMBER_TYPE, STRING_TYPE], NUMBER_TYPE),
  'three-parameter function'
);

// Function application tests
const contextWithIdentity = { identity: createFunctionType([NUMBER_TYPE], NUMBER_TYPE) };
assertEquals(
  testSynthesize('identity(42)', contextWithIdentity),
  NUMBER_TYPE,
  'function application'
);

// Inline function application with type annotation
assertEquals(
  testSynthesize('((x: number) => x)(42)'),
  NUMBER_TYPE,
  'inline function application'
);

// Binary expressions
assertEquals(testSynthesize('10 + 20'), NUMBER_TYPE, 'number addition');
assertEquals(testSynthesize('10 - 5'), NUMBER_TYPE, 'number subtraction');
assertEquals(testSynthesize('10 * 5'), NUMBER_TYPE, 'number multiplication');
assertEquals(testSynthesize('10 / 2'), NUMBER_TYPE, 'number division');
assertEquals(testSynthesize('"hello" + "world"'), STRING_TYPE, 'string concatenation');

// Conditional expressions
assertEquals(testSynthesize('true ? 10 : 20'), NUMBER_TYPE, 'conditional with numbers');
assertEquals(testSynthesize('false ? "yes" : "no"'), STRING_TYPE, 'conditional with strings');

// Check function tests
runTest(() => {
  testCheck('true', BOOL_TYPE);
  return 'check boolean literal';
});

runTest(() => {
  testCheck('42', NUMBER_TYPE);
  return 'check number literal';
});

runTest(() => {
  testCheck('(x: number) => x + 1', createFunctionType([NUMBER_TYPE], NUMBER_TYPE));
  return 'check function type';
});

// Error cases
runFail(() => testCheck('true', NUMBER_TYPE)); // wrong type
runFail(() => testSynthesize('unknownVar')); // undefined variable
runFail(() => testSynthesize('10 + "hello"')); // type mismatch in addition
runFail(() => testSynthesize('true ? 10 : "hello"')); // conditional branch type mismatch
runFail(() => testSynthesize('"hello" ? 1 : 2')); // non-boolean condition

// Test the problematic case: unannotated lambda in function application
console.log('\nTesting the problematic case from the original question:');
runFail(() => testSynthesize('((b) => b ? false : true)(true)'));

// But it works with type annotation:
assertEquals(
  testSynthesize('((b: boolean) => b ? false : true)(true)'),
  BOOL_TYPE,
  'annotated lambda in application works'
);

// Multi-parameter function applications
const contextWithAdd = { add: createFunctionType([NUMBER_TYPE, NUMBER_TYPE], NUMBER_TYPE) };
assertEquals(
  testSynthesize('add(10, 20)', contextWithAdd),
  NUMBER_TYPE,
  'two-argument function application'
);

const contextWithThreeParam = { fn: createFunctionType([NUMBER_TYPE, STRING_TYPE, BOOL_TYPE], NUMBER_TYPE) };
assertEquals(
  testSynthesize('fn(42, "hello", true)', contextWithThreeParam),
  NUMBER_TYPE,
  'three-argument function application'
);

// Inline multi-parameter function application
assertEquals(
  testSynthesize('((x: number, y: number) => x + y)(10, 20)'),
  NUMBER_TYPE,
  'inline two-parameter function application'
);

// Higher-order functions
assertEquals(
  testSynthesize('(f: (x: number) => number) => (x: number) => f(x)'),
  createFunctionType(
    [createFunctionType([NUMBER_TYPE], NUMBER_TYPE)],
    createFunctionType([NUMBER_TYPE], NUMBER_TYPE)
  ),
  'higher-order function'
);

// Complex nested expression
assertEquals(
  testSynthesize('((f: (x: number) => number) => f(10))((x: number) => x + 1)'),
  NUMBER_TYPE,
  'complex nested function application'
);

// Test error cases for multi-parameter functions
runFail(() => testSynthesize('add(10)', contextWithAdd)); // too few arguments
runFail(() => testSynthesize('add(10, 20, 30)', contextWithAdd)); // too many arguments
runFail(() => testSynthesize('add(10, "hello")', contextWithAdd)); // wrong argument type

// Generic function tests
console.log('\n🧬 Testing generic functions:');

// Generic identity function
const T = createTypeVariable('T');
assertEquals(
  testSynthesize('(x: T) => x'),
  createFunctionType([T], T),
  'generic identity function'
);

// Generic function application with type inference
const contextWithGenericId = { 
  identity: createFunctionType([createTypeVariable('T')], createTypeVariable('T'))
};

assertEquals(
  testSynthesize('identity(42)', contextWithGenericId),
  NUMBER_TYPE,
  'generic identity applied to number'
);

assertEquals(
  testSynthesize('identity("hello")', contextWithGenericId),
  STRING_TYPE,
  'generic identity applied to string'
);

assertEquals(
  testSynthesize('identity(true)', contextWithGenericId),
  BOOL_TYPE,
  'generic identity applied to boolean'
);

// Generic function with multiple type variables
const U = createTypeVariable('U');
const V = createTypeVariable('V');
const contextWithGenericPair = {
  first: createFunctionType([U, V], U)
};

assertEquals(
  testSynthesize('first(42, "hello")', contextWithGenericPair),
  NUMBER_TYPE,
  'generic first function extracts first type'
);

assertEquals(
  testSynthesize('first("world", true)', contextWithGenericPair),
  STRING_TYPE,
  'generic first function with different types'
);

// Inline generic function application
assertEquals(
  testSynthesize('((x: T) => x)(42)'),
  NUMBER_TYPE,
  'inline generic function application'
);

// Higher-order generic function
const contextWithGenericMap = {
  map: createFunctionType(
    [createFunctionType([createTypeVariable('A')], createTypeVariable('B')), createTypeVariable('A')],
    createTypeVariable('B')
  )
};

const contextWithIncrement = {
  ...contextWithGenericMap,
  increment: createFunctionType([NUMBER_TYPE], NUMBER_TYPE)
};

assertEquals(
  testSynthesize('map(increment, 5)', contextWithIncrement),
  NUMBER_TYPE,
  'higher-order generic function application'
);

console.log('\n✅ All tests passed!');

// Variable Declaration Type Inference Tests
console.log('\n📦 Testing Variable Declaration Type Inference...\n');

// Test 1: Simple number inference
console.log('Test 1: let x = 4');
const test1 = processProgram('let x = 4');
assertEquals(test1.context.x, NUMBER_TYPE, 'x should be inferred as number');
console.log('✓ Variable x inferred as number\n');

// Test 2: String inference
console.log('Test 2: let message = "hello"');
const test2 = processProgram('let message = "hello"');
assertEquals(test2.context.message, STRING_TYPE, 'message should be inferred as string');
console.log('✓ Variable message inferred as string\n');

// Test 3: Boolean inference
console.log('Test 3: let flag = true');
const test3 = processProgram('let flag = true');
assertEquals(test3.context.flag, BOOL_TYPE, 'flag should be inferred as boolean');
console.log('✓ Variable flag inferred as boolean\n');

// Test 4: Multiple variables
console.log('Test 4: Multiple variable declarations');
const test4 = processProgram(`
  let a = 10
  let b = "world"
  let c = false
`);
assertEquals(test4.context.a, NUMBER_TYPE, 'a should be number');
assertEquals(test4.context.b, STRING_TYPE, 'b should be string');
assertEquals(test4.context.c, BOOL_TYPE, 'c should be boolean');
console.log('✓ Multiple variables inferred correctly\n');

// Test 5: Using variables after declaration
console.log('Test 5: Using variables after declaration');
const test5 = processProgram(`
  let x = 5
  let y = 10
  x + y
`);
assertEquals(test5.context.x, NUMBER_TYPE, 'x should be number');
assertEquals(test5.context.y, NUMBER_TYPE, 'y should be number');
// The last expression should synthesize to number
const lastResult = test5.results[test5.results.length - 1];
assertEquals(lastResult.type, NUMBER_TYPE, 'x + y should be number');
console.log('✓ Variables can be used in expressions\n');

// Test 6: Inference from expressions
console.log('Test 6: Inference from expressions');
const test6 = processProgram(`
  let sum = 10 + 20
  let concat = "hello" + "world"
  let expr = true ? 1 : 2
`);
assertEquals(test6.context.sum, NUMBER_TYPE, 'sum should be number');
assertEquals(test6.context.concat, STRING_TYPE, 'concat should be string');
assertEquals(test6.context.expr, NUMBER_TYPE, 'expr should be number');
console.log('✓ Types inferred from complex expressions\n');

// Test 7: Function type inference
console.log('Test 7: Function type inference');
const test7 = processProgram(`
  let identity = (x: number) => x
  let constant = (x: number) => true
`);
assertEquals(
  test7.context.identity, 
  createFunctionType([NUMBER_TYPE], NUMBER_TYPE),
  'identity should be (number) => number'
);
assertEquals(
  test7.context.constant,
  createFunctionType([NUMBER_TYPE], BOOL_TYPE),
  'constant should be (number) => boolean'
);
console.log('✓ Function types inferred correctly\n');

// Test 8: Using inferred function variables
console.log('Test 8: Using inferred function variables');
const test8 = processProgram(`
  let addOne = (x: number) => x + 1
  addOne(5)
`);
assertEquals(
  test8.context.addOne,
  createFunctionType([NUMBER_TYPE], NUMBER_TYPE),
  'addOne should be (number) => number'
);
const funcCallResult = test8.results[test8.results.length - 1];
assertEquals(funcCallResult.type, NUMBER_TYPE, 'addOne(5) should return number');
console.log('✓ Inferred functions can be called\n');

// Test 9: Chained variable usage
console.log('Test 9: Chained variable usage');
const test9 = processProgram(`
  let a = 100
  let b = a
  let c = b + 50
`);
assertEquals(test9.context.a, NUMBER_TYPE, 'a should be number');
assertEquals(test9.context.b, NUMBER_TYPE, 'b should be number (from a)');
assertEquals(test9.context.c, NUMBER_TYPE, 'c should be number');
console.log('✓ Variables can reference other variables\n');

// Test 10: Explicit type annotations still work
console.log('Test 10: Explicit type annotations');
const test10 = processProgram(`
  let x: number = 42
  let y: string = "test"
  let z: boolean = true
`);
assertEquals(test10.context.x, NUMBER_TYPE, 'x should be number (explicit)');
assertEquals(test10.context.y, STRING_TYPE, 'y should be string (explicit)');
assertEquals(test10.context.z, BOOL_TYPE, 'z should be boolean (explicit)');
console.log('✓ Explicit type annotations respected\n');

// Test 11: const declarations work the same
console.log('Test 11: const declarations');
const test11 = processProgram(`
  const PI = 3.14
  const NAME = "TypeScript"
  const ENABLED = true
`);
assertEquals(test11.context.PI, NUMBER_TYPE, 'PI should be number');
assertEquals(test11.context.NAME, STRING_TYPE, 'NAME should be string');
assertEquals(test11.context.ENABLED, BOOL_TYPE, 'ENABLED should be boolean');
console.log('✓ const declarations work with inference\n');

// Test 12: Comparison operators
console.log('Test 12: Comparison operators');
const test12 = processProgram(`
  let greater = 10 > 5
  let equal = "a" === "a"
  let notEqual = true !== false
`);
assertEquals(test12.context.greater, BOOL_TYPE, 'greater should be boolean');
assertEquals(test12.context.equal, BOOL_TYPE, 'equal should be boolean');
assertEquals(test12.context.notEqual, BOOL_TYPE, 'notEqual should be boolean');
console.log('✓ Comparison operators infer boolean type\n');

// Test 13: var declarations work too
console.log('Test 13: var declarations');
const test13 = processProgram(`
  var oldStyle = 42
  var legacy = "still works"
`);
assertEquals(test13.context.oldStyle, NUMBER_TYPE, 'var declaration inferred as number');
assertEquals(test13.context.legacy, STRING_TYPE, 'var declaration inferred as string');
console.log('✓ var declarations work with inference\n');

// Error cases for variable declarations
console.log('Testing variable declaration error cases...');

// Should fail: variable without initializer or type
runFail(() => {
  processProgram('let x');
});
console.log('✓ Fails when variable has no initializer or type');

// Should fail: type mismatch with explicit annotation
runFail(() => {
  processProgram('let x: string = 42');
});
console.log('✓ Fails when initializer doesn\'t match explicit type\n');

console.log('✅ All variable declaration tests passed!');

// Test complex function with local variables
console.log('\n🎯 Testing Complex Function with Local Variables...\n');

console.log('Test: Function with type signature and inferred locals');
const complexFunctionTest = processProgram(`
  let calculatePrice = (basePrice: number, taxRate: number) => {
    let discount = 0.1
    let discountedPrice = basePrice * (1 - discount)
    let tax = discountedPrice * taxRate
    let finalPrice = discountedPrice + tax
    let isExpensive = finalPrice > 100
    let message = "Price: $" + "calculated"
    finalPrice
  }
`);

// Check the function type
assertEquals(
  complexFunctionTest.context.calculatePrice,
  createFunctionType([NUMBER_TYPE, NUMBER_TYPE], NUMBER_TYPE),
  'calculatePrice should be (number, number) => number'
);

console.log('✓ Function type signature correct: (number, number) => number\n');

// Test calling this function to ensure it works end-to-end
const functionCallTest = processProgram(`
  let calculatePrice = (basePrice: number, taxRate: number) => {
    let discount = 0.1
    let discountedPrice = basePrice * (1 - discount)
    let tax = discountedPrice * taxRate
    let finalPrice = discountedPrice + tax
    let isExpensive = finalPrice > 100
    let message = "Price: $" + "calculated"
    finalPrice
  }
  calculatePrice(50, 0.08)
`);

// Check that the function call returns the correct type
const callResult = functionCallTest.results[functionCallTest.results.length - 1];
assertEquals(callResult.type, NUMBER_TYPE, 'Function call should return number');
console.log('✓ Function call returns correct type\n');

// Test more complex example with nested functions
console.log('Test: Nested functions with local variables');
const nestedFunctionTest = processProgram(`
  let processOrder = (items: number, unitPrice: number) => {
    let subtotal = items * unitPrice
    let calculateTax = (amount: number) => {
      let rate = 0.075
      let tax = amount * rate
      tax
    }
    let tax = calculateTax(subtotal)
    let shipping = items > 5 ? 0 : 10
    let total = subtotal + tax + shipping
    let formattedTotal = total + 0.005
    total
  }
`);

assertEquals(
  nestedFunctionTest.context.processOrder,
  createFunctionType([NUMBER_TYPE, NUMBER_TYPE], NUMBER_TYPE),
  'processOrder should be (number, number) => number'
);
console.log('✓ Nested function with locals works correctly\n');

// Test function that returns a function (higher-order)
console.log('Test: Higher-order function with local variables');
const higherOrderTest = processProgram(`
  let createMultiplier = (factor: number) => {
    let cachedFactor = factor
    let multiplier = (x: number) => {
      let adjusted = x + 1
      let result = adjusted * cachedFactor
      result
    }
    multiplier
  }
`);

assertEquals(
  higherOrderTest.context.createMultiplier,
  createFunctionType(
    [NUMBER_TYPE], 
    createFunctionType([NUMBER_TYPE], NUMBER_TYPE)
  ),
  'createMultiplier should be (number) => (number) => number'
);
console.log('✓ Higher-order function with locals works correctly\n');

// Test function with mixed return types that should fail
console.log('Test: Function with inconsistent return type should fail');
runFail(() => {
  processProgram(`
    let badFunction = (x: number) => {
      let result = x > 0
      let message = "positive"
      result ? message : 42
    }
  `);
});
console.log('✓ Function with mixed return types correctly fails\n');

// Test very complex realistic example
console.log('Test: Complex realistic data processing function');
const complexRealisticTest = processProgram(`
  let processUserData = (userId: number, age: number, active: boolean) => {
    let isAdult = age >= 18
    let userCategory = isAdult ? "adult" : "minor"
    let baseScore = age * 2
    let activityBonus = active ? 10 : 0
    let finalScore = baseScore + activityBonus
    let isPremium = finalScore > 50
    let discount = isPremium ? 0.2 : 0.0
    let categoryMultiplier = userCategory === "adult" ? 1.5 : 1.0
    let adjustedScore = finalScore * categoryMultiplier
    adjustedScore
  }
`);

assertEquals(
  complexRealisticTest.context.processUserData,
  createFunctionType([NUMBER_TYPE, NUMBER_TYPE, BOOL_TYPE], NUMBER_TYPE),
  'processUserData should be (number, number, boolean) => number'
);
console.log('✓ Complex realistic function works correctly\n');

console.log('✅ All complex function tests passed!');
console.log('\n📝 Variable Declaration Summary:');
console.log('- Variables can be declared without type annotations');
console.log('- Types are automatically inferred from initializers');
console.log('- let, const, and var all support type inference');
console.log('- Inferred variables can be used in subsequent expressions');
console.log('- Explicit type annotations are still supported when needed');
console.log('- Variables can reference other variables in their initializers');
console.log('- Complex functions with many locals work correctly');
console.log('- Function bodies can contain multiple variable declarations');
console.log('- Local variables are properly scoped within function bodies');