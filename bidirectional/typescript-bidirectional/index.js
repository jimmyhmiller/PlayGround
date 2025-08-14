import ts from 'typescript';

// Parse TypeScript source code into AST
const parseTypeScript = (code) => {
  return ts.createSourceFile(
    'temp.ts',
    code,
    ts.ScriptTarget.Latest,
    true
  );
};

// Helper functions to work with TypeScript AST
const isLiteralExpression = (node) => {
  return ts.isNumericLiteral(node) || 
         ts.isStringLiteral(node) || 
         node.kind === ts.SyntaxKind.TrueKeyword || 
         node.kind === ts.SyntaxKind.FalseKeyword;
};

const isIdentifier = (node) => ts.isIdentifier(node);
const isArrowFunction = (node) => ts.isArrowFunction(node);
const isFunctionExpression = (node) => ts.isFunctionExpression(node);
const isCallExpression = (node) => ts.isCallExpression(node);
const isBinaryExpression = (node) => ts.isBinaryExpression(node);
const isConditionalExpression = (node) => ts.isConditionalExpression(node);

// Type environment for variables
class TypeContext {
  constructor(parent = null) {
    this.parent = parent;
    this.bindings = new Map();
  }

  lookup(name) {
    if (this.bindings.has(name)) {
      return this.bindings.get(name);
    }
    if (this.parent) {
      return this.parent.lookup(name);
    }
    throw new Error(`Variable ${name} not found in context`);
  }

  extend(name, type) {
    const newContext = new TypeContext(this);
    newContext.bindings.set(name, type);
    return newContext;
  }
}

// Type representation
const createFunctionType = (paramType, returnType) => ({
  kind: 'function',
  paramType,
  returnType
});

const createPrimitiveType = (name) => ({
  kind: 'primitive',
  name
});

const BOOL_TYPE = createPrimitiveType('boolean');
const NUMBER_TYPE = createPrimitiveType('number');
const STRING_TYPE = createPrimitiveType('string');

// Type equality check
const typesEqual = (type1, type2) => {
  if (type1.kind !== type2.kind) return false;
  
  switch (type1.kind) {
    case 'primitive':
      return type1.name === type2.name;
    case 'function':
      return typesEqual(type1.paramType, type2.paramType) && 
             typesEqual(type1.returnType, type2.returnType);
    default:
      return false;
  }
};

// Convert TypeScript type annotation to our type representation
const convertTypeAnnotation = (typeNode) => {
  if (!typeNode) return null;
  
  switch (typeNode.kind) {
    case ts.SyntaxKind.BooleanKeyword:
      return BOOL_TYPE;
    case ts.SyntaxKind.NumberKeyword:
      return NUMBER_TYPE;
    case ts.SyntaxKind.StringKeyword:
      return STRING_TYPE;
    case ts.SyntaxKind.FunctionType:
      const paramType = convertTypeAnnotation(typeNode.parameters[0]?.type);
      const returnType = convertTypeAnnotation(typeNode.type);
      return createFunctionType(paramType, returnType);
    default:
      throw new Error(`Unsupported type annotation: ${ts.SyntaxKind[typeNode.kind]}`);
  }
};

// Synthesize type for an expression
const synthesize = (node, context) => {
  switch (node.kind) {
    case ts.SyntaxKind.TrueKeyword:
    case ts.SyntaxKind.FalseKeyword:
      return BOOL_TYPE;
      
    case ts.SyntaxKind.NumericLiteral:
      return NUMBER_TYPE;
      
    case ts.SyntaxKind.StringLiteral:
      return STRING_TYPE;
      
    case ts.SyntaxKind.Identifier:
      return context.lookup(node.text);
      
    case ts.SyntaxKind.ArrowFunction:
    case ts.SyntaxKind.FunctionExpression:
      const param = node.parameters[0];
      if (!param || !param.type) {
        throw new Error("Can't determine type. Please add type annotation to function parameter");
      }
      
      const paramType = convertTypeAnnotation(param.type);
      const extendedContext = context.extend(param.name.text, paramType);
      const bodyType = synthesize(node.body, extendedContext);
      
      return createFunctionType(paramType, bodyType);
      
    case ts.SyntaxKind.CallExpression:
      const functionType = synthesize(node.expression, context);
      if (functionType.kind !== 'function') {
        throw new Error(`Expected function type, got ${functionType.kind}`);
      }
      
      const argType = synthesize(node.arguments[0], context);
      if (!typesEqual(functionType.paramType, argType)) {
        throw new Error(`Expected argument of type ${JSON.stringify(functionType.paramType)}, got ${JSON.stringify(argType)}`);
      }
      
      return functionType.returnType;
      
    case ts.SyntaxKind.BinaryExpression:
      const left = synthesize(node.left, context);
      const right = synthesize(node.right, context);
      
      switch (node.operatorToken.kind) {
        case ts.SyntaxKind.PlusToken:
          if (typesEqual(left, NUMBER_TYPE) && typesEqual(right, NUMBER_TYPE)) {
            return NUMBER_TYPE;
          }
          if (typesEqual(left, STRING_TYPE) && typesEqual(right, STRING_TYPE)) {
            return STRING_TYPE;
          }
          throw new Error(`+ requires matching operand types (number+number or string+string)`);
          
        case ts.SyntaxKind.MinusToken:
        case ts.SyntaxKind.AsteriskToken:
        case ts.SyntaxKind.SlashToken:
          if (!typesEqual(left, NUMBER_TYPE) || !typesEqual(right, NUMBER_TYPE)) {
            throw new Error(`${ts.tokenToString(node.operatorToken.kind)} requires number operands`);
          }
          return NUMBER_TYPE;
          
        default:
          throw new Error(`Unsupported binary operator: ${ts.SyntaxKind[node.operatorToken.kind]}`);
      }
      
    case ts.SyntaxKind.ConditionalExpression:
      const conditionType = synthesize(node.condition, context);
      if (!typesEqual(conditionType, BOOL_TYPE)) {
        throw new Error(`Conditional expression condition must be boolean`);
      }
      
      const thenType = synthesize(node.whenTrue, context);
      const elseType = synthesize(node.whenFalse, context);
      
      if (!typesEqual(thenType, elseType)) {
        throw new Error(`Conditional expression branches must have same type`);
      }
      
      return thenType;
      
    case ts.SyntaxKind.ParenthesizedExpression:
      return synthesize(node.expression, context);
      
    default:
      throw new Error(`Unsupported expression: ${ts.SyntaxKind[node.kind]}`);
  }
};

// Check that an expression has the expected type
const check = (node, expectedType, context) => {
  switch (node.kind) {
    case ts.SyntaxKind.ArrowFunction:
    case ts.SyntaxKind.FunctionExpression:
      if (expectedType.kind !== 'function') {
        throw new Error(`Expected function type, got ${expectedType.kind}`);
      }
      
      const param = node.parameters[0];
      const paramType = param.type ? convertTypeAnnotation(param.type) : expectedType.paramType;
      
      if (!typesEqual(paramType, expectedType.paramType)) {
        throw new Error(`Parameter type mismatch`);
      }
      
      const extendedContext = context.extend(param.name.text, paramType);
      check(node.body, expectedType.returnType, extendedContext);
      return;
      
    case ts.SyntaxKind.ParenthesizedExpression:
      check(node.expression, expectedType, context);
      return;
      
    default:
      const actualType = synthesize(node, context);
      if (!typesEqual(actualType, expectedType)) {
        throw new Error(`Expected type ${JSON.stringify(expectedType)}, got ${JSON.stringify(actualType)}`);
      }
  }
};

// Helper function to get the first statement/expression from parsed code
const getExpression = (sourceFile) => {
  const statement = sourceFile.statements[0];
  if (ts.isExpressionStatement(statement)) {
    return statement.expression;
  }
  return statement;
};

// Test runner utilities
const runTest = (testCase, shouldFail = false) => {
  try {
    const result = testCase();
    console.log('✓', result);
    if (shouldFail) {
      console.error('✗ Should have failed, but got:', result);
    }
    return result;
  } catch (e) {
    if (shouldFail) {
      console.log('✓ Expected failure:', e.message);
      return true;
    }
    console.error('✗ Unexpected failure:', e.message);
    throw e;
  }
};

const runFail = (testCase) => runTest(testCase, true);

const assertEquals = (actual, expected, testName = '') => {
  const actualStr = JSON.stringify(actual);
  const expectedStr = JSON.stringify(expected);
  if (actualStr === expectedStr) {
    console.log(`✓ ${testName}: ${expectedStr}`);
    return actual;
  } else {
    console.error(`✗ ${testName}: expected ${expectedStr}, got ${actualStr}`);
    throw new Error(`Assertion failed: expected ${expectedStr}, got ${actualStr}`);
  }
};

// Export for testing
export { 
  parseTypeScript, 
  synthesize, 
  check, 
  TypeContext, 
  getExpression,
  runTest,
  runFail, 
  assertEquals,
  BOOL_TYPE,
  NUMBER_TYPE,
  STRING_TYPE,
  createFunctionType
};