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
  createFunctionType,
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

// Basic literal tests
assertEquals(testInfer('true'), BOOL_TYPE, 'boolean literal');
assertEquals(testInfer('false'), BOOL_TYPE, 'boolean literal false');
assertEquals(testInfer('42'), NUMBER_TYPE, 'number literal');
assertEquals(testInfer('"hello"'), STRING_TYPE, 'string literal');

// Variable tests
const contextWithVar = { x: BOOL_TYPE };
assertEquals(testInfer('x', contextWithVar), BOOL_TYPE, 'boolean variable');

// Function tests - with type annotations
assertEquals(
  testInfer('(x: number) => x'),
  createFunctionType([NUMBER_TYPE], NUMBER_TYPE),
  'identity function with annotation'
);

assertEquals(
  testInfer('(x: number) => true'),
  createFunctionType([NUMBER_TYPE], BOOL_TYPE),
  'function number -> boolean'
);

// Multi-parameter function tests
assertEquals(
  testInfer('(x: number, y: string) => x + 42'),
  createFunctionType([NUMBER_TYPE, STRING_TYPE], NUMBER_TYPE),
  'two-parameter function'
);

assertEquals(
  testInfer('(a: number, b: number, c: string) => a + b'),
  createFunctionType([NUMBER_TYPE, NUMBER_TYPE, STRING_TYPE], NUMBER_TYPE),
  'three-parameter function'
);

// Function application tests
const contextWithIdentity = { identity: createFunctionType([NUMBER_TYPE], NUMBER_TYPE) };
assertEquals(
  testInfer('identity(42)', contextWithIdentity),
  NUMBER_TYPE,
  'function application'
);

// Inline function application with type annotation
assertEquals(
  testInfer('((x: number) => x)(42)'),
  NUMBER_TYPE,
  'inline function application'
);

// Binary expressions
assertEquals(testInfer('10 + 20'), NUMBER_TYPE, 'number addition');
assertEquals(testInfer('10 - 5'), NUMBER_TYPE, 'number subtraction');
assertEquals(testInfer('10 * 5'), NUMBER_TYPE, 'number multiplication');
assertEquals(testInfer('10 / 2'), NUMBER_TYPE, 'number division');
assertEquals(testInfer('"hello" + "world"'), STRING_TYPE, 'string concatenation');

// Conditional expressions
assertEquals(testInfer('true ? 10 : 20'), NUMBER_TYPE, 'conditional with numbers');
assertEquals(testInfer('false ? "yes" : "no"'), STRING_TYPE, 'conditional with strings');

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
runFail(() => testInfer('unknownVar')); // undefined variable
runFail(() => testInfer('10 + "hello"')); // type mismatch in addition
runFail(() => testInfer('true ? 10 : "hello"')); // conditional branch type mismatch
runFail(() => testInfer('"hello" ? 1 : 2')); // non-boolean condition

// Test the problematic case: unannotated lambda in function application
console.log('\nTesting the problematic case from the original question:');
runFail(() => testInfer('((b) => b ? false : true)(true)'));

// But it works with type annotation:
assertEquals(
  testInfer('((b: boolean) => b ? false : true)(true)'),
  BOOL_TYPE,
  'annotated lambda in application works'
);

// Multi-parameter function applications
const contextWithAdd = { add: createFunctionType([NUMBER_TYPE, NUMBER_TYPE], NUMBER_TYPE) };
assertEquals(
  testInfer('add(10, 20)', contextWithAdd),
  NUMBER_TYPE,
  'two-argument function application'
);

const contextWithThreeParam = { fn: createFunctionType([NUMBER_TYPE, STRING_TYPE, BOOL_TYPE], NUMBER_TYPE) };
assertEquals(
  testInfer('fn(42, "hello", true)', contextWithThreeParam),
  NUMBER_TYPE,
  'three-argument function application'
);

// Inline multi-parameter function application
assertEquals(
  testInfer('((x: number, y: number) => x + y)(10, 20)'),
  NUMBER_TYPE,
  'inline two-parameter function application'
);

// Higher-order functions
assertEquals(
  testInfer('(f: (x: number) => number) => (x: number) => f(x)'),
  createFunctionType(
    [createFunctionType([NUMBER_TYPE], NUMBER_TYPE)],
    createFunctionType([NUMBER_TYPE], NUMBER_TYPE)
  ),
  'higher-order function'
);

// Complex nested expression
assertEquals(
  testInfer('((f: (x: number) => number) => f(10))((x: number) => x + 1)'),
  NUMBER_TYPE,
  'complex nested function application'
);

// Test error cases for multi-parameter functions
runFail(() => testInfer('add(10)', contextWithAdd)); // too few arguments
runFail(() => testInfer('add(10, 20, 30)', contextWithAdd)); // too many arguments
runFail(() => testInfer('add(10, "hello")', contextWithAdd)); // wrong argument type

// Generic function tests
console.log('\n🧬 Testing generic functions:');

// Generic identity function
const T = createTypeVariable('T');
assertEquals(
  testInfer('(x: T) => x'),
  createFunctionType([T], T),
  'generic identity function'
);

// Generic function application with type inference
const contextWithGenericId = { 
  identity: createFunctionType([createTypeVariable('T')], createTypeVariable('T'))
};

assertEquals(
  testInfer('identity(42)', contextWithGenericId),
  NUMBER_TYPE,
  'generic identity applied to number'
);

assertEquals(
  testInfer('identity("hello")', contextWithGenericId),
  STRING_TYPE,
  'generic identity applied to string'
);

assertEquals(
  testInfer('identity(true)', contextWithGenericId),
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
  testInfer('first(42, "hello")', contextWithGenericPair),
  NUMBER_TYPE,
  'generic first function extracts first type'
);

assertEquals(
  testInfer('first("world", true)', contextWithGenericPair),
  STRING_TYPE,
  'generic first function with different types'
);

// Inline generic function application
assertEquals(
  testInfer('((x: T) => x)(42)'),
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
  testInfer('map(increment, 5)', contextWithIncrement),
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
// The last expression should infer to number
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

// Now allowed: variable without initializer (creates unknown type until assigned)
const uninitializedTest = processProgram('let x');
assertEquals(uninitializedTest.context.x.kind, 'unknown', 'uninitialized variable should be unknown');
console.log('✓ Uninitialized variables are allowed (marked as unknown)');

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

// Test return type annotations
console.log('\n🎯 Testing Return Type Annotations...\n');

console.log('Test: Function with explicit return type');
const returnTypeTest = processProgram(`
  let createMultiplier = (factor: number): (x: number) => number => {
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
  returnTypeTest.context.createMultiplier,
  createFunctionType(
    [NUMBER_TYPE], 
    createFunctionType([NUMBER_TYPE], NUMBER_TYPE)
  ),
  'createMultiplier should be (number) => (number) => number with explicit return type'
);
console.log('✓ Function with explicit return type works correctly\n');

// Test simple return type annotation
console.log('Test: Simple function with return type annotation');
const simpleReturnType = processProgram(`
  let addOne = (x: number): number => x + 1
`);

assertEquals(
  simpleReturnType.context.addOne,
  createFunctionType([NUMBER_TYPE], NUMBER_TYPE),
  'addOne should be (number) => number'
);
console.log('✓ Simple return type annotation works\n');

// Test that return type mismatch fails
console.log('Test: Return type mismatch should fail');
runFail(() => {
  processProgram(`
    let badFunction = (x: number): string => x + 1
  `);
});
console.log('✓ Return type mismatch correctly fails\n');

// Test return type with boolean
console.log('Test: Return type with boolean');
const booleanReturnType = processProgram(`
  let isPositive = (x: number): boolean => x > 0
`);

assertEquals(
  booleanReturnType.context.isPositive,
  createFunctionType([NUMBER_TYPE], BOOL_TYPE),
  'isPositive should be (number) => boolean'
);
console.log('✓ Boolean return type annotation works\n');

console.log('✅ All return type annotation tests passed!');

// Deferred Type Inference Tests
console.log('\n🔮 Testing Deferred Type Inference (Assignment-Based)...\n');

// Test 1: Basic deferred inference
console.log('Test 1: Declare then assign');
const deferredTest1 = processProgram(`
  let x
  x = 42
`);
assertEquals(deferredTest1.context.x, NUMBER_TYPE, 'x should be inferred as number after assignment');
console.log('✓ Variable type inferred from first assignment\n');

// Test 2: Multiple variables with deferred inference
console.log('Test 2: Multiple variables with deferred inference');
const deferredTest2 = processProgram(`
  let a
  let b
  let c
  a = 100
  b = "hello"
  c = true
`);
assertEquals(deferredTest2.context.a, NUMBER_TYPE, 'a should be number');
assertEquals(deferredTest2.context.b, STRING_TYPE, 'b should be string');
assertEquals(deferredTest2.context.c, BOOL_TYPE, 'c should be boolean');
console.log('✓ Multiple variables inferred from assignments\n');

// Test 3: Using variables after assignment
console.log('Test 3: Using variables after assignment');
const deferredTest3 = processProgram(`
  let x
  let y
  x = 10
  y = 20
  x + y
`);
assertEquals(deferredTest3.context.x, NUMBER_TYPE, 'x should be number');
assertEquals(deferredTest3.context.y, NUMBER_TYPE, 'y should be number');
const lastExpr = deferredTest3.results[deferredTest3.results.length - 1];
assertEquals(lastExpr.type, NUMBER_TYPE, 'x + y should be number');
console.log('✓ Variables can be used after type inference\n');

// Test 4: Complex expressions in assignments
console.log('Test 4: Complex expressions in assignments');
const deferredTest4 = processProgram(`
  let result
  let message
  let flag
  result = 10 + 5 * 2
  message = "Value: " + "computed"
  flag = result > 15
`);
assertEquals(deferredTest4.context.result, NUMBER_TYPE, 'result should be number');
assertEquals(deferredTest4.context.message, STRING_TYPE, 'message should be string');
assertEquals(deferredTest4.context.flag, BOOL_TYPE, 'flag should be boolean');
console.log('✓ Complex expressions infer correct types\n');

// Test 5: Function assignment with deferred inference
console.log('Test 5: Function assignment with deferred inference');
const deferredTest5 = processProgram(`
  let myFunc
  myFunc = (x: number) => x * 2
`);
assertEquals(
  deferredTest5.context.myFunc,
  createFunctionType([NUMBER_TYPE], NUMBER_TYPE),
  'myFunc should be (number) => number'
);
console.log('✓ Function type inferred from assignment\n');

// Test 6: Chain of assignments
console.log('Test 6: Chain of assignments');
const deferredTest6 = processProgram(`
  let first
  let second
  first = 100
  second = first
`);
assertEquals(deferredTest6.context.first, NUMBER_TYPE, 'first should be number');
assertEquals(deferredTest6.context.second, NUMBER_TYPE, 'second should be number (from first)');
console.log('✓ Variables can be assigned from other variables\n');

// Error cases
console.log('Testing deferred inference error cases...');

// Should fail: using variable before assignment
runFail(() => {
  processProgram(`
    let x
    x + 1
  `);
});
console.log('✓ Fails when using variable before assignment');

// Should fail: inconsistent assignment
runFail(() => {
  processProgram(`
    let x
    x = 42
    x = "hello"
  `);
});
console.log('✓ Fails when assignment type is inconsistent');

// Should fail: assignment to undeclared variable
runFail(() => {
  processProgram(`
    y = 42
  `);
});
console.log('✓ Fails when assigning to undeclared variable\n');

// Test 7: Mixed declaration styles
console.log('Test 7: Mixed declaration and assignment styles');
const mixedTest = processProgram(`
  let initialized = 10
  let deferred
  deferred = initialized + 5
  let another = deferred * 2
`);
assertEquals(mixedTest.context.initialized, NUMBER_TYPE, 'initialized should be number');
assertEquals(mixedTest.context.deferred, NUMBER_TYPE, 'deferred should be number');
assertEquals(mixedTest.context.another, NUMBER_TYPE, 'another should be number');
console.log('✓ Mixed styles work together\n');

// Test 8: Complex function with deferred inference inside
console.log('Test 8: Function with deferred inference inside');
const deferredInFunction = processProgram(`
  let processor = (input: number) => {
    let temp
    let result
    temp = input * 2
    result = temp + 10
    result
  }
`);
assertEquals(
  deferredInFunction.context.processor,
  createFunctionType([NUMBER_TYPE], NUMBER_TYPE),
  'processor should be (number) => number'
);
console.log('✓ Deferred inference works inside functions\n');

console.log('✅ All deferred type inference tests passed!');

// Advanced Edge Case Tests for Deferred Inference
console.log('\n⚡ Testing Advanced Edge Cases...\n');

// Test 1: Variable never assigned - should fail when used
console.log('Test 1: Variable never assigned');
runFail(() => {
  processProgram(`
    let x
    let y = x + 1
  `);
});
console.log('✓ Fails when variable used without assignment');

// Test 2: Multiple assignments to same variable (consistency)
console.log('Test 2: Multiple consistent assignments');
const consistentTest = processProgram(`
  let x
  x = 10
  x = 20
  x = 30
`);
assertEquals(consistentTest.context.x, NUMBER_TYPE, 'x should remain number');
console.log('✓ Multiple consistent assignments work\n');

// Test 3: Assignment from conditional expression works
console.log('Test 3: Assignment from conditional expression');
const conditionalExprTest = processProgram(`
  let result
  result = true ? 42 : 100
`);
assertEquals(conditionalExprTest.context.result, NUMBER_TYPE, 'result should be number');
console.log('✓ Conditional expression assignment works');

// Test 4: Assignment from conditional with different types should fail
console.log('Test 4: Conditional with different types should fail');
runFail(() => {
  processProgram(`
    let result
    result = true ? 42 : "hello"
  `);
});
console.log('✓ Conditional with mismatched types fails\n');

// Test 5: Complex assignment chains
console.log('Test 5: Complex assignment dependency chains');
const chainTest = processProgram(`
  let a
  let b  
  let c
  a = 10
  b = a * 2
  c = b + a
`);
assertEquals(chainTest.context.a, NUMBER_TYPE, 'a should be number');
assertEquals(chainTest.context.b, NUMBER_TYPE, 'b should be number');
assertEquals(chainTest.context.c, NUMBER_TYPE, 'c should be number');
console.log('✓ Complex dependency chains work\n');

// Test 6: Function parameter shadowing
console.log('Test 6: Function parameter shadowing deferred variables');
const shadowTest = processProgram(`
  let x
  let fn = (x: string) => {
    let y
    y = x + " processed"
    y
  }
  x = 42
`);
assertEquals(shadowTest.context.x, NUMBER_TYPE, 'outer x should be number');
assertEquals(
  shadowTest.context.fn,
  createFunctionType([STRING_TYPE], STRING_TYPE),
  'fn should be (string) => string'
);
console.log('✓ Parameter shadowing works correctly\n');

// Test 7: Assignment ordering matters
console.log('Test 7: Assignment ordering validation');
runFail(() => {
  processProgram(`
    let x
    let y = x  // Error: x not assigned yet
    x = 42
  `);
});
console.log('✓ Fails when using variable before it\'s assigned');

// Test 8: Reassignment type checking
console.log('Test 8: Reassignment type consistency');
runFail(() => {
  processProgram(`
    let counter
    counter = 0
    counter = counter + 1  // This should work
    counter = "done"       // This should fail
  `);
});
console.log('✓ Fails on inconsistent reassignment\n');

// Current limitations (documented for future implementation)
console.log('📋 Current Limitations (by design for safety):');
console.log('• if/else statements not supported (would require control flow analysis)');
console.log('• Arrays/objects not yet supported (would require structural typing)');
console.log('• Loop constructs not supported (would require flow analysis)');
console.log('• This prevents complex assignment scenarios that could be unsafe\n');

// Show what conditional expressions DO work
console.log('✓ Conditional expressions (ternary) ARE supported:');
const ternaryExample = processProgram(`
  let value
  value = true ? 100 : 200
`);
console.log(`✓ value: ${formatType(ternaryExample.context.value)} (from ternary)\n`);

console.log('✅ All edge case tests passed!');

// Control Flow Tests with If Statements
console.log('\n🌊 Testing Control Flow with If Statements...\n');

// Test 1: Same type assigned in both branches
console.log('Test 1: Same type in both if/else branches');
const sameTypeTest = processProgram(`
  let x
  if (true) {
    x = 42
  } else {
    x = 100
  }
  x + 1
`);
assertEquals(sameTypeTest.context.x, NUMBER_TYPE, 'x should be number from both branches');
console.log('✓ Same type in both branches works\n');

// Test 2: Different types should fail
console.log('Test 2: Different types in if/else branches should fail');
runFail(() => {
  processProgram(`
    let x
    if (true) {
      x = 42
    } else {
      x = "hello"
    }
  `);
});
console.log('✓ Different types in branches correctly fails');

// Test 3: Assignment in then branch only (no else)
console.log('Test 3: Assignment in then branch only should fail');
runFail(() => {
  processProgram(`
    let x
    if (true) {
      x = 42
    }
  `);
});
console.log('✓ Assignment in then-only branch correctly fails\n');

// Test 4: Assignment in else but not then
console.log('Test 4: Assignment in else but not then should fail');
runFail(() => {
  processProgram(`
    let x
    if (true) {
      // x not assigned here
    } else {
      x = 42
    }
  `);
});
console.log('✓ Unbalanced assignment fails');

// Test 5: Multiple variables with different assignment patterns
console.log('Test 5: Multiple variables with complex patterns');
const multiVarTest = processProgram(`
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
assertEquals(multiVarTest.context.a, NUMBER_TYPE, 'a should be number');
assertEquals(multiVarTest.context.b, NUMBER_TYPE, 'b should be number');
assertEquals(multiVarTest.context.c, NUMBER_TYPE, 'c should be number');
console.log('✓ Multiple variables with if/else work\n');

// Test 6: Nested if statements
console.log('Test 6: Nested if statements');
const nestedTest = processProgram(`
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
assertEquals(nestedTest.context.x, NUMBER_TYPE, 'x should be number');
assertEquals(nestedTest.context.y, NUMBER_TYPE, 'y should be number');
console.log('✓ Nested if statements work\n');

// Test 7: If with function calls and complex expressions
console.log('Test 7: If with complex expressions');
const complexIfTest = processProgram(`
  let result
  let processor = (n: number) => n * 2
  if (processor(5) > 8) {
    result = 100
  } else {
    result = 200
  }
`);
assertEquals(complexIfTest.context.result, NUMBER_TYPE, 'result should be number');
console.log('✓ If with complex condition expressions work\n');

// Test 8: Block statements vs single statements
console.log('Test 8: Block vs single statement branches');
runFail(() => {
  processProgram(`
    let x
    let y
    if (true) {
      x = 10
      y = 20
    } else
      x = 30
  `);
});
console.log('✓ Block vs single statement handling works\n');

console.log('✅ All control flow tests passed!');
console.log('\n📝 Complete Variable Declaration Summary:');
console.log('- Variables can be declared without type annotations or initializers');
console.log('- Types are inferred from first assignment when not initialized');
console.log('- Subsequent assignments must be consistent with inferred type');
console.log('- Variables cannot be used before their type is determined');
console.log('- let, const, and var all support deferred type inference');
console.log('- Mixed initialization and assignment styles work together');
console.log('- Deferred inference works within function bodies');
console.log('- Complex expressions and function assignments work correctly');
console.log('- Assignment ordering is validated for safety');
console.log('- Conditional expressions (ternary) work for assignments');
console.log('- If/else statements with consistent types across branches work');
console.log('- Variables assigned in all branches are properly typed');
console.log('- Variables not assigned in all branches remain unknown');
console.log('- Type conflicts across branches are detected and prevented');