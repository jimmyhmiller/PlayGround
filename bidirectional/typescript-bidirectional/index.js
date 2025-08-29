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

// Simple context helpers
const lookupVariable = (context, name) => {
  if (context[name] === undefined) {
    throw new Error(`Variable ${name} not found in context`);
  }
  return context[name];
};

// Type representation
const createFunctionType = (paramTypes, returnType) => {
  if (!Array.isArray(paramTypes)) {
    throw new Error('paramTypes must be an array');
  }
  return {
    kind: 'function',
    paramTypes,
    returnType
  };
};

const createPrimitiveType = (name) => ({
  kind: 'primitive',
  name
});

const createTypeVariable = (name) => ({
  kind: 'variable',
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
    case 'variable':
      return type1.name === type2.name;
    case 'function':
      if (type1.paramTypes.length !== type2.paramTypes.length) return false;
      return type1.paramTypes.every((param1, i) => typesEqual(param1, type2.paramTypes[i])) &&
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
    case ts.SyntaxKind.TypeReference:
      // Handle generic type variables like T, U, etc.
      if (ts.isIdentifier(typeNode.typeName)) {
        return createTypeVariable(typeNode.typeName.text);
      }
      throw new Error(`Unsupported type reference: ${typeNode.typeName}`);
    case ts.SyntaxKind.FunctionType:
      const paramTypes = typeNode.parameters.map(param => convertTypeAnnotation(param.type));
      const returnType = convertTypeAnnotation(typeNode.type);
      return createFunctionType(paramTypes, returnType);
    default:
      throw new Error(`Unsupported type annotation: ${ts.SyntaxKind[typeNode.kind]}`);
  }
};

// Type variable utilities
const isTypeVariable = (type) => type.kind === 'variable';

const collectTypeVars = (type) => {
  switch (type.kind) {
    case 'variable':
      return [type.name];
    case 'function':
      return [...new Set([
        ...type.paramTypes.flatMap(collectTypeVars),
        ...collectTypeVars(type.returnType)
      ])];
    default:
      return [];
  }
};

const substitute = (type, substitutions) => {
  switch (type.kind) {
    case 'variable':
      return substitutions[type.name] || type;
    case 'function':
      return createFunctionType(
        type.paramTypes.map(param => substitute(param, substitutions)),
        substitute(type.returnType, substitutions)
      );
    default:
      return type;
  }
};

const unify = (type1, type2, substitutions = {}) => {
  type1 = substitute(type1, substitutions);
  type2 = substitute(type2, substitutions);
  
  if (typesEqual(type1, type2)) {
    return substitutions;
  } else if (isTypeVariable(type1)) {
    return { ...substitutions, [type1.name]: type2 };
  } else if (isTypeVariable(type2)) {
    return { ...substitutions, [type2.name]: type1 };
  } else if (type1.kind === 'function' && type2.kind === 'function') {
    if (type1.paramTypes.length !== type2.paramTypes.length) {
      throw new Error(`Cannot unify function types with different arity`);
    }
    
    let newSubs = substitutions;
    for (let i = 0; i < type1.paramTypes.length; i++) {
      newSubs = unify(type1.paramTypes[i], type2.paramTypes[i], newSubs);
    }
    newSubs = unify(type1.returnType, type2.returnType, newSubs);
    return newSubs;
  } else {
    throw new Error(`Cannot unify ${JSON.stringify(type1)} with ${JSON.stringify(type2)}`);
  }
};

const isGenericFunctionType = (type) => {
  return type.kind === 'function' && collectTypeVars(type).length > 0;
};

// Helper function to format types nicely
const formatType = (type) => {
  switch (type.kind) {
    case 'primitive':
      return type.name;
    case 'function':
      const paramStr = type.paramTypes.map(formatType).join(', ');
      return `(${paramStr}) => ${formatType(type.returnType)}`;
    case 'variable':
      return type.name;
    default:
      return JSON.stringify(type);
  }
};

// Synthesize the type of a function body (could be block or expression)
const synthesizeFunctionBody = (bodyNode, context) => {
  if (ts.isBlock(bodyNode)) {
    // Handle block statement - process all statements and return the type of the last expression
    const { context: finalContext, results } = processStatements(bodyNode.statements, context);
    
    // Find the last non-declaration statement's type
    for (let i = results.length - 1; i >= 0; i--) {
      if (results[i].kind === 'expression') {
        return results[i].type;
      }
    }
    
    throw new Error('Function body must end with an expression');
  } else {
    // Handle expression body
    return synthesize(bodyNode, context);
  }
};

// Process variable declarations and build context
const processVariableDeclaration = (node, context) => {
  if (node.kind !== ts.SyntaxKind.VariableDeclaration) {
    throw new Error('Expected variable declaration');
  }
  
  const varName = node.name.text;
  
  // If there's an explicit type annotation, use it
  if (node.type) {
    const annotatedType = convertTypeAnnotation(node.type);
    
    // If there's also an initializer, check that it matches the annotation
    if (node.initializer) {
      const initializerType = synthesize(node.initializer, context);
      if (!typesEqual(annotatedType, initializerType)) {
        throw new Error(`Variable ${varName} annotated as ${formatType(annotatedType)} but initialized with ${formatType(initializerType)}`);
      }
    }
    
    return { ...context, [varName]: annotatedType };
  }
  
  // Otherwise, infer the type from the initializer
  if (!node.initializer) {
    throw new Error(`Variable ${varName} needs either a type annotation or an initializer`);
  }
  
  const inferredType = synthesize(node.initializer, context);
  return { ...context, [varName]: inferredType };
};

// Process a statement and return updated context
const processStatement = (statement, context) => {
  switch (statement.kind) {
    case ts.SyntaxKind.VariableStatement:
      // Process each declaration in the statement
      let newContext = context;
      for (const declaration of statement.declarationList.declarations) {
        newContext = processVariableDeclaration(declaration, newContext);
      }
      return newContext;
      
    case ts.SyntaxKind.ExpressionStatement:
      // For expression statements, just synthesize but don't update context
      synthesize(statement.expression, context);
      return context;
      
    default:
      throw new Error(`Unsupported statement: ${ts.SyntaxKind[statement.kind]}`);
  }
};

// Process multiple statements and build context incrementally
const processStatements = (statements, initialContext = {}) => {
  let context = initialContext;
  const results = [];
  
  for (const statement of statements) {
    if (statement.kind === ts.SyntaxKind.VariableStatement) {
      context = processStatement(statement, context);
      // Store the variable declarations for reporting
      for (const declaration of statement.declarationList.declarations) {
        const varName = declaration.name.text;
        results.push({ 
          kind: 'variable', 
          name: varName, 
          type: context[varName] 
        });
      }
    } else if (statement.kind === ts.SyntaxKind.ExpressionStatement) {
      const type = synthesize(statement.expression, context);
      results.push({ 
        kind: 'expression', 
        type 
      });
    } else {
      context = processStatement(statement, context);
    }
  }
  
  return { context, results };
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
      return lookupVariable(context, node.text);
      
    case ts.SyntaxKind.ArrowFunction:
    case ts.SyntaxKind.FunctionExpression:
      if (node.parameters.some(param => !param.type)) {
        throw new Error("Can't determine type. Please add type annotation to function parameter");
      }
      
      const paramTypes = node.parameters.map(param => convertTypeAnnotation(param.type));
      let extendedContext = { ...context };
      
      node.parameters.forEach((param, i) => {
        extendedContext[param.name.text] = paramTypes[i];
      });
      
      const bodyType = synthesizeFunctionBody(node.body, extendedContext);
      
      return createFunctionType(paramTypes, bodyType);
      
    case ts.SyntaxKind.CallExpression:
      const functionType = synthesize(node.expression, context);
      if (functionType.kind !== 'function') {
        throw new Error(`Expected function type, got ${functionType.kind}`);
      }
      
      if (node.arguments.length !== functionType.paramTypes.length) {
        throw new Error(`Expected ${functionType.paramTypes.length} arguments, got ${node.arguments.length}`);
      }
      
      if (isGenericFunctionType(functionType)) {
        // Handle generic function application with type inference
        let substitutions = {};
        
        for (let i = 0; i < node.arguments.length; i++) {
          const argType = synthesize(node.arguments[i], context);
          substitutions = unify(functionType.paramTypes[i], argType, substitutions);
        }
        
        const instantiatedReturnType = substitute(functionType.returnType, substitutions);
        return instantiatedReturnType;
      } else {
        // Handle non-generic function application
        for (let i = 0; i < node.arguments.length; i++) {
          const argType = synthesize(node.arguments[i], context);
          if (!typesEqual(functionType.paramTypes[i], argType)) {
            throw new Error(`Expected argument ${i} of type ${JSON.stringify(functionType.paramTypes[i])}, got ${JSON.stringify(argType)}`);
          }
        }
        
        return functionType.returnType;
      }
      
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
        case ts.SyntaxKind.PercentToken:
          if (!typesEqual(left, NUMBER_TYPE) || !typesEqual(right, NUMBER_TYPE)) {
            throw new Error(`${ts.tokenToString(node.operatorToken.kind)} requires number operands`);
          }
          return NUMBER_TYPE;
          
        case ts.SyntaxKind.GreaterThanToken:
        case ts.SyntaxKind.LessThanToken:
        case ts.SyntaxKind.GreaterThanEqualsToken:
        case ts.SyntaxKind.LessThanEqualsToken:
        case ts.SyntaxKind.EqualsEqualsToken:
        case ts.SyntaxKind.EqualsEqualsEqualsToken:
        case ts.SyntaxKind.ExclamationEqualsToken:
        case ts.SyntaxKind.ExclamationEqualsEqualsToken:
          // Comparison operators return boolean
          if (typesEqual(left, NUMBER_TYPE) && typesEqual(right, NUMBER_TYPE)) {
            return BOOL_TYPE;
          }
          if (typesEqual(left, STRING_TYPE) && typesEqual(right, STRING_TYPE)) {
            return BOOL_TYPE;
          }
          if (typesEqual(left, BOOL_TYPE) && typesEqual(right, BOOL_TYPE)) {
            return BOOL_TYPE;
          }
          throw new Error(`Comparison requires matching operand types`);
          
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
      
      if (node.parameters.length !== expectedType.paramTypes.length) {
        throw new Error(`Expected ${expectedType.paramTypes.length} parameters, got ${node.parameters.length}`);
      }
      
      let extendedContext = { ...context };
      
      for (let i = 0; i < node.parameters.length; i++) {
        const param = node.parameters[i];
        const paramType = param.type ? convertTypeAnnotation(param.type) : expectedType.paramTypes[i];
        
        if (!typesEqual(paramType, expectedType.paramTypes[i])) {
          throw new Error(`Parameter ${i} type mismatch`);
        }
        
        extendedContext[param.name.text] = paramType;
      }
      
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

// Helper to process a full program (multiple statements)
const processProgram = (code, initialContext = {}) => {
  const sourceFile = parseTypeScript(code);
  return processStatements(sourceFile.statements, initialContext);
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
  getExpression,
  processProgram,
  processVariableDeclaration,
  processStatement,
  processStatements,
  synthesizeFunctionBody,
  formatType,
  runTest,
  runFail, 
  assertEquals,
  BOOL_TYPE,
  NUMBER_TYPE,
  STRING_TYPE,
  createFunctionType,
  createTypeVariable,
  createPrimitiveType
};