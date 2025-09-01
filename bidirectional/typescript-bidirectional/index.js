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

const createUnknownType = (name) => ({
  kind: 'unknown',
  name
});

// Merge two contexts, handling conflicts for unknown types
const mergeContexts = (context1, context2, conflictHandler) => {
  const merged = { ...context1 };
  
  for (const [varName, type2] of Object.entries(context2)) {
    const type1 = context1[varName];
    
    if (type1 === undefined) {
      merged[varName] = type2;
    } else if (type1.kind === 'unknown' && type2.kind !== 'unknown') {
      // Type inferred in context2 but not context1 - use conflict handler if available
      if (conflictHandler) {
        merged[varName] = conflictHandler(varName, type1, type2);
      } else {
        merged[varName] = type2; // Default: use the concrete type
      }
    } else if (type1.kind !== 'unknown' && type2.kind === 'unknown') {
      // Type inferred in context1 but not context2 - use conflict handler if available
      if (conflictHandler) {
        merged[varName] = conflictHandler(varName, type1, type2);
      } else {
        merged[varName] = type1; // Default: keep the concrete type
      }
    } else if (!typesEqual(type1, type2)) {
      if (conflictHandler) {
        merged[varName] = conflictHandler(varName, type1, type2);
      } else {
        throw new Error(`Type conflict for variable ${varName}: ${formatType(type1)} vs ${formatType(type2)}`);
      }
    } else {
      merged[varName] = type1; // Same type, keep it
    }
  }
  
  return merged;
};

// Check if a variable is definitely assigned (not unknown) in a context
const isDefinitelyAssigned = (context, varName) => {
  const varType = context[varName];
  return varType !== undefined && varType.kind !== 'unknown';
};

const BOOL_TYPE = createPrimitiveType('boolean');
const NUMBER_TYPE = createPrimitiveType('number');
const STRING_TYPE = createPrimitiveType('string');
const VOID_TYPE = createPrimitiveType('void');

// Type equality check
const typesEqual = (type1, type2) => {
  if (type1.kind !== type2.kind) return false;
  
  switch (type1.kind) {
    case 'primitive':
      return type1.name === type2.name;
    case 'variable':
      return type1.name === type2.name;
    case 'unknown':
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

const substitute = (type, substitutions = {}) => {
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
    case 'unknown':
      return `unknown(${type.name})`;
    default:
      return JSON.stringify(type);
  }
};

// Infer the type of a function body (could be block or expression)
const inferFunctionBody = (bodyNode, context) => {
  if (ts.isBlock(bodyNode)) {
    // Handle block statement - look for return statements
    const { context: finalContext, results } = processStatements(bodyNode.statements, context);
    
    // Find and validate return statement types
    let returnType = null;
    let hasReturn = false;
    
    for (let i = 0; i < results.length; i++) {
      if (results[i].kind === 'return') {
        if (!hasReturn) {
          // First return statement - set the expected type
          returnType = results[i].type;
          hasReturn = true;
        } else {
          // Subsequent return statements must match the first
          if (!typesEqual(results[i].type, returnType)) {
            throw new Error(`All return statements must have the same type. Expected ${formatType(returnType)}, got ${formatType(results[i].type)}`);
          }
        }
      }
    }
    
    // If no return statement found, this is a void function
    return hasReturn ? returnType : VOID_TYPE;
  } else {
    // Handle expression body (arrow functions)
    return infer(bodyNode, context);
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
      const initializerType = infer(node.initializer, context);
      if (!typesEqual(annotatedType, initializerType)) {
        throw new Error(`Variable ${varName} annotated as ${formatType(annotatedType)} but initialized with ${formatType(initializerType)}`);
      }
    }
    
    return { ...context, [varName]: annotatedType };
  }
  
  // If there's an initializer, infer the type from it
  if (node.initializer) {
    const inferredType = infer(node.initializer, context);
    return { ...context, [varName]: inferredType };
  }
  
  // No type annotation and no initializer - mark as unknown for later inference
  return { ...context, [varName]: createUnknownType(varName) };
};

// Process if statements with control flow analysis
const processIfStatement = (ifStatement, context) => {
  // Type check the condition
  const conditionType = infer(ifStatement.expression, context);
  if (!typesEqual(conditionType, BOOL_TYPE)) {
    throw new Error(`If condition must be boolean, got ${formatType(conditionType)}`);
  }
  
  // Process the then branch
  const thenResult = ifStatement.thenStatement.kind === ts.SyntaxKind.Block 
    ? processStatements(ifStatement.thenStatement.statements, context)
    : { context: processStatement(ifStatement.thenStatement, context), results: [] };
  
  // Process the else branch if it exists
  let elseResult = { context, results: [] };
  if (ifStatement.elseStatement) {
    elseResult = ifStatement.elseStatement.kind === ts.SyntaxKind.Block
      ? processStatements(ifStatement.elseStatement.statements, context)
      : { context: processStatement(ifStatement.elseStatement, context), results: [] };
  }
  
  // Collect all results from both branches
  const allResults = [...thenResult.results, ...elseResult.results];
  
  // Merge the contexts with type consistency checking
  const finalContext = mergeContexts(thenResult.context, elseResult.context, (varName, thenType, elseType) => {
    // If both branches assign the same type, use it
    if (typesEqual(thenType, elseType)) {
      return thenType;
    }
    
    // Handle unbalanced assignments
    if (thenType.kind === 'unknown' && elseType.kind !== 'unknown') {
      if (ifStatement.elseStatement) {
        throw new Error(`Variable ${varName} assigned in else branch but not then branch`);
      } else {
        // No else clause, variable not assigned in then - keep unknown
        return thenType;
      }
    }
    
    if (elseType.kind === 'unknown' && thenType.kind !== 'unknown') {
      if (ifStatement.elseStatement) {
        throw new Error(`Variable ${varName} assigned in then branch but not else branch`);
      } else {
        throw new Error(`Variable ${varName} assigned in then branch but not on all paths (missing else branch)`);
      }
    }
    
    // Different concrete types - this is an error
    throw new Error(`Variable ${varName} assigned different types: ${formatType(thenType)} vs ${formatType(elseType)}`);
  });
  
  return { context: finalContext, results: allResults };
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
      
    case ts.SyntaxKind.FunctionDeclaration:
      // Process function declaration and add it to context
      const functionName = statement.name.text;
      const functionType = infer(statement, context);
      return { ...context, [functionName]: functionType };
      
    case ts.SyntaxKind.ReturnStatement:
      // Return statements don't modify context, they're handled by function body processing
      return context;
      
    case ts.SyntaxKind.ExpressionStatement:
      // Check if this is an assignment that could infer a type
      const expr = statement.expression;
      if (ts.isBinaryExpression(expr) && expr.operatorToken.kind === ts.SyntaxKind.FirstAssignment) {
        return processAssignment(expr, context);
      }
      
      // For other expression statements, just infer but don't update context
      infer(statement.expression, context);
      return context;
      
    case ts.SyntaxKind.IfStatement:
      return processIfStatement(statement, context).context;
      
    default:
      throw new Error(`Unsupported statement: ${ts.SyntaxKind[statement.kind]}`);
  }
};

// Process assignment expressions and infer types for unknown variables
const processAssignment = (node, context) => {
  if (!ts.isBinaryExpression(node) || node.operatorToken.kind !== ts.SyntaxKind.FirstAssignment) {
    throw new Error('Expected assignment expression');
  }
  
  if (!ts.isIdentifier(node.left)) {
    throw new Error('Left side of assignment must be an identifier');
  }
  
  const varName = node.left.text;
  const rightType = infer(node.right, context);
  
  // Check if variable exists in context
  if (context[varName] === undefined) {
    throw new Error(`Variable ${varName} not declared`);
  }
  
  const currentType = context[varName];
  
  if (currentType.kind === 'unknown') {
    // First assignment to an unknown variable - infer its type
    return { ...context, [varName]: rightType };
  } else {
    // Variable already has a type - check consistency
    if (!typesEqual(currentType, rightType)) {
      throw new Error(`Cannot assign ${formatType(rightType)} to variable ${varName} of type ${formatType(currentType)}`);
    }
    return context;
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
    } else if (statement.kind === ts.SyntaxKind.FunctionDeclaration) {
      context = processStatement(statement, context);
      // Store the function declaration for reporting
      const functionName = statement.name.text;
      results.push({ 
        kind: 'function', 
        name: functionName, 
        type: context[functionName] 
      });
    } else if (statement.kind === ts.SyntaxKind.ReturnStatement) {
      // Process return statement
      if (statement.expression) {
        const returnType = infer(statement.expression, context);
        results.push({
          kind: 'return',
          type: returnType
        });
      } else {
        // Return with no expression (void)
        results.push({
          kind: 'return',
          type: { kind: 'primitive', name: 'void' }
        });
      }
      context = processStatement(statement, context);
    } else if (statement.kind === ts.SyntaxKind.ExpressionStatement) {
      const expr = statement.expression;
      // Check if this is an assignment
      if (ts.isBinaryExpression(expr) && expr.operatorToken.kind === ts.SyntaxKind.FirstAssignment) {
        context = processAssignment(expr, context);
        results.push({ 
          kind: 'assignment',
          variable: expr.left.text,
          type: context[expr.left.text]
        });
      } else {
        const type = infer(statement.expression, context);
        results.push({ 
          kind: 'expression', 
          type 
        });
      }
    } else if (statement.kind === ts.SyntaxKind.IfStatement) {
      const ifResult = processIfStatement(statement, context);
      context = ifResult.context;
      results.push(...ifResult.results);
    } else {
      context = processStatement(statement, context);
    }
  }
  
  return { context, results };
};

// Infer type for an expression
const infer = (node, context) => {
  switch (node.kind) {
    case ts.SyntaxKind.TrueKeyword:
    case ts.SyntaxKind.FalseKeyword:
      return BOOL_TYPE;
      
    case ts.SyntaxKind.NumericLiteral:
      return NUMBER_TYPE;
      
    case ts.SyntaxKind.StringLiteral:
      return STRING_TYPE;
      
    case ts.SyntaxKind.Identifier:
      const varType = context[node.text];
      if (varType === undefined) {
        throw new Error(`Variable ${node.text} not found in context`);
      }
      if (varType.kind === 'unknown') {
        throw new Error(`Variable ${node.text} used before assignment`);
      }
      return varType;
      
    case ts.SyntaxKind.ArrowFunction:
    case ts.SyntaxKind.FunctionExpression:
    case ts.SyntaxKind.FunctionDeclaration:
      if (node.parameters.some(param => !param.type)) {
        throw new Error("Can't determine type. Please add type annotation to function parameter");
      }
      
      const paramTypes = node.parameters.map(param => convertTypeAnnotation(param.type));
      let extendedContext = { ...context };
      
      node.parameters.forEach((param, i) => {
        extendedContext[param.name.text] = paramTypes[i];
      });
      
      // Check if there's an explicit return type annotation
      if (node.type) {
        const explicitReturnType = convertTypeAnnotation(node.type);
        // Verify the body matches the declared return type
        const bodyType = inferFunctionBody(node.body, extendedContext);
        if (!typesEqual(bodyType, explicitReturnType)) {
          throw new Error(`Function body returns ${formatType(bodyType)} but declared return type is ${formatType(explicitReturnType)}`);
        }
        return createFunctionType(paramTypes, explicitReturnType);
      } else {
        // No explicit return type - infer from body
        const bodyType = inferFunctionBody(node.body, extendedContext);
        return createFunctionType(paramTypes, bodyType);
      }
      
    case ts.SyntaxKind.CallExpression:
      const functionType = infer(node.expression, context);
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
          const argType = infer(node.arguments[i], context);
          substitutions = unify(functionType.paramTypes[i], argType, substitutions);
        }
        
        const instantiatedReturnType = substitute(functionType.returnType, substitutions);
        return instantiatedReturnType;
      } else {
        // Handle non-generic function application
        for (let i = 0; i < node.arguments.length; i++) {
          const argType = infer(node.arguments[i], context);
          if (!typesEqual(functionType.paramTypes[i], argType)) {
            throw new Error(`Expected argument ${i} of type ${JSON.stringify(functionType.paramTypes[i])}, got ${JSON.stringify(argType)}`);
          }
        }
        
        return functionType.returnType;
      }
      
    case ts.SyntaxKind.BinaryExpression:
      // Handle assignment separately (this shouldn't happen in expression context)
      if (node.operatorToken.kind === ts.SyntaxKind.FirstAssignment) {
        throw new Error('Assignment should be handled at statement level, not expression level');
      }
      
      const left = infer(node.left, context);
      const right = infer(node.right, context);
      
      switch (node.operatorToken.kind) {
        case ts.SyntaxKind.PlusToken:
          // Number + Number = Number (arithmetic)
          if (typesEqual(left, NUMBER_TYPE) && typesEqual(right, NUMBER_TYPE)) {
            return NUMBER_TYPE;
          }
          // String + String = String (concatenation)
          if (typesEqual(left, STRING_TYPE) && typesEqual(right, STRING_TYPE)) {
            return STRING_TYPE;
          }
          // String + Number = String (concatenation with coercion)
          if (typesEqual(left, STRING_TYPE) && typesEqual(right, NUMBER_TYPE)) {
            return STRING_TYPE;
          }
          // Number + String = String (concatenation with coercion)
          if (typesEqual(left, NUMBER_TYPE) && typesEqual(right, STRING_TYPE)) {
            return STRING_TYPE;
          }
          throw new Error(`+ requires operands of types (number, number) or (string, string) or (string, number) or (number, string)`);
          
        case ts.SyntaxKind.MinusToken:
        case ts.SyntaxKind.AsteriskToken:
        case ts.SyntaxKind.SlashToken:
        case ts.SyntaxKind.PercentToken:
          if (!typesEqual(left, NUMBER_TYPE) || !typesEqual(right, NUMBER_TYPE)) {
            throw new Error(`${ts.SyntaxKind[node.operatorToken.kind]} requires number operands`);
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
      const conditionType = infer(node.condition, context);
      if (!typesEqual(conditionType, BOOL_TYPE)) {
        throw new Error(`Conditional expression condition must be boolean`);
      }
      
      const thenType = infer(node.whenTrue, context);
      const elseType = infer(node.whenFalse, context);
      
      if (!typesEqual(thenType, elseType)) {
        throw new Error(`Conditional expression branches must have same type`);
      }
      
      return thenType;
      
    case ts.SyntaxKind.ParenthesizedExpression:
      return infer(node.expression, context);
      
    default:
      throw new Error(`Unsupported expression: ${ts.SyntaxKind[node.kind]}`);
  }
};

// Check that an expression has the expected type
const check = (node, expectedType, context) => {
  switch (node.kind) {
    case ts.SyntaxKind.ArrowFunction:
    case ts.SyntaxKind.FunctionExpression:
    case ts.SyntaxKind.FunctionDeclaration:
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
      
      // Check if there's an explicit return type annotation
      const returnTypeToCheck = node.type ? convertTypeAnnotation(node.type) : expectedType.returnType;
      
      if (!typesEqual(returnTypeToCheck, expectedType.returnType)) {
        throw new Error(`Function declares return type ${formatType(returnTypeToCheck)} but expected ${formatType(expectedType.returnType)}`);
      }
      
      if (ts.isBlock(node.body)) {
        // Handle block statement body
        const { context: finalContext, results } = processStatements(node.body.statements, extendedContext);
        
        // Check ALL return statements have the correct type
        let foundReturn = false;
        for (let i = 0; i < results.length; i++) {
          if (results[i].kind === 'return') {
            foundReturn = true;
            if (!typesEqual(results[i].type, returnTypeToCheck)) {
              throw new Error(`Expected function return type ${formatType(returnTypeToCheck)}, got ${formatType(results[i].type)}`);
            }
          }
        }
        
        // If no return statement found, check if this should be void
        if (!foundReturn && !typesEqual(returnTypeToCheck, VOID_TYPE)) {
          throw new Error(`Expected function return type ${formatType(returnTypeToCheck)}, but function has no return statement (void)`);
        }
      } else {
        // Handle expression body
        check(node.body, returnTypeToCheck, extendedContext);
      }
      return;
      
    case ts.SyntaxKind.ParenthesizedExpression:
      check(node.expression, expectedType, context);
      return;
      
    default:
      const actualType = infer(node, context);
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
  infer, 
  check, 
  getExpression,
  processProgram,
  processVariableDeclaration,
  processStatement,
  processStatements,
  processAssignment,
  inferFunctionBody,
  formatType,
  runTest,
  runFail, 
  assertEquals,
  BOOL_TYPE,
  NUMBER_TYPE,
  STRING_TYPE,
  VOID_TYPE,
  createFunctionType, // Still needed internally by parseType helper
  createTypeVariable,
  createPrimitiveType,
  createUnknownType
};