import { 
  parseTypeScript, 
  synthesize, 
  check, 
  getExpression,
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