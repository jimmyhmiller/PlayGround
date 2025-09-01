const ts = require('typescript');

// Type system
const createPrimitiveType = (name) => ({ kind: 'primitive', name });
const createFunctionType = (paramTypes, returnType) => ({ kind: 'function', paramTypes, returnType });

// Basic types
const NUMBER_TYPE = createPrimitiveType('number');
const STRING_TYPE = createPrimitiveType('string');
const BOOLEAN_TYPE = createPrimitiveType('boolean');
const VOID_TYPE = createPrimitiveType('void');

// Type equality
const typesEqual = (type1, type2) => {
  if (type1.kind !== type2.kind) return false;
  
  if (type1.kind === 'primitive') {
    return type1.name === type2.name;
  }
  
  if (type1.kind === 'function') {
    if (type1.paramTypes.length !== type2.paramTypes.length) return false;
    for (let i = 0; i < type1.paramTypes.length; i++) {
      if (!typesEqual(type1.paramTypes[i], type2.paramTypes[i])) return false;
    }
    return typesEqual(type1.returnType, type2.returnType);
  }
  
  return false;
};

// Convert TypeScript type annotations to our internal types
const convertTypeAnnotation = (typeNode) => {
  switch (typeNode.kind) {
    case ts.SyntaxKind.NumberKeyword:
      return NUMBER_TYPE;
    case ts.SyntaxKind.StringKeyword:
      return STRING_TYPE;
    case ts.SyntaxKind.BooleanKeyword:
      return BOOLEAN_TYPE;
    case ts.SyntaxKind.VoidKeyword:
      return VOID_TYPE;
    default:
      throw new Error(`Unsupported type: ${ts.SyntaxKind[typeNode.kind]}`);
  }
};

// Format types for display
const formatType = (type) => {
  if (type.kind === 'primitive') {
    return type.name;
  }
  if (type.kind === 'function') {
    const params = type.paramTypes.map(formatType).join(', ');
    return `(${params}) => ${formatType(type.returnType)}`;
  }
  return 'unknown';
};

// Type inference
const infer = (node, context) => {
  switch (node.kind) {
    case ts.SyntaxKind.NumericLiteral:
      return NUMBER_TYPE;
      
    case ts.SyntaxKind.StringLiteral:
      return STRING_TYPE;
      
    case ts.SyntaxKind.TrueKeyword:
    case ts.SyntaxKind.FalseKeyword:
      return BOOLEAN_TYPE;
      
    case ts.SyntaxKind.Identifier:
      const varName = node.text;
      if (context[varName]) {
        return context[varName];
      }
      throw new Error(`Variable '${varName}' is not defined`);
      
    case ts.SyntaxKind.BinaryExpression:
      const left = infer(node.left, context);
      const right = infer(node.right, context);
      
      switch (node.operatorToken.kind) {
        case ts.SyntaxKind.PlusToken:
          // Number + Number = Number
          if (typesEqual(left, NUMBER_TYPE) && typesEqual(right, NUMBER_TYPE)) {
            return NUMBER_TYPE;
          }
          // String + String = String
          if (typesEqual(left, STRING_TYPE) && typesEqual(right, STRING_TYPE)) {
            return STRING_TYPE;
          }
          // String + Number = String (concatenation)
          if (typesEqual(left, STRING_TYPE) && typesEqual(right, NUMBER_TYPE)) {
            return STRING_TYPE;
          }
          // Number + String = String (concatenation)
          if (typesEqual(left, NUMBER_TYPE) && typesEqual(right, STRING_TYPE)) {
            return STRING_TYPE;
          }
          throw new Error('Invalid operands for + operator');
          
        case ts.SyntaxKind.MinusToken:
        case ts.SyntaxKind.AsteriskToken:
        case ts.SyntaxKind.SlashToken:
          if (!typesEqual(left, NUMBER_TYPE) || !typesEqual(right, NUMBER_TYPE)) {
            throw new Error('Arithmetic requires number operands');
          }
          return NUMBER_TYPE;
          
        case ts.SyntaxKind.GreaterThanToken:
          if (!typesEqual(left, NUMBER_TYPE) || !typesEqual(right, NUMBER_TYPE)) {
            throw new Error('Comparison requires number operands');
          }
          return BOOLEAN_TYPE;
          
        default:
          throw new Error(`Unsupported operator: ${ts.SyntaxKind[node.operatorToken.kind]}`);
      }
      
    case ts.SyntaxKind.CallExpression:
      const funcType = infer(node.expression, context);
      if (funcType.kind !== 'function') {
        throw new Error('Cannot call non-function');
      }
      
      // Check argument count
      if (node.arguments.length !== funcType.paramTypes.length) {
        throw new Error(`Expected ${funcType.paramTypes.length} arguments, got ${node.arguments.length}`);
      }
      
      // Check argument types
      for (let i = 0; i < node.arguments.length; i++) {
        const argType = infer(node.arguments[i], context);
        if (!typesEqual(argType, funcType.paramTypes[i])) {
          throw new Error(`Argument ${i} expects ${formatType(funcType.paramTypes[i])}, got ${formatType(argType)}`);
        }
      }
      
      return funcType.returnType;
      
    default:
      throw new Error(`Unsupported expression: ${ts.SyntaxKind[node.kind]}`);
  }
};

// Process statements and build context
const processStatements = (statements, context) => {
  const newContext = { ...context };
  const results = [];
  
  for (const stmt of statements) {
    const result = processStatement(stmt, newContext);
    if (result) {
      if (result.kind === 'if-results') {
        // Flatten results from if statements
        results.push(...result.results);
      } else {
        results.push(result);
      }
    }
  }
  
  return { context: newContext, results };
};

// Process individual statement
const processStatement = (node, context) => {
  switch (node.kind) {
    case ts.SyntaxKind.VariableStatement:
      for (const declaration of node.declarationList.declarations) {
        if (declaration.initializer) {
          const inferredType = infer(declaration.initializer, context);
          context[declaration.name.text] = inferredType;
        }
      }
      return null;
      
    case ts.SyntaxKind.ReturnStatement:
      if (node.expression) {
        const returnType = infer(node.expression, context);
        return { kind: 'return', type: returnType };
      }
      return { kind: 'return', type: VOID_TYPE };
      
    case ts.SyntaxKind.IfStatement:
      // Process condition
      infer(node.expression, context);
      
      // Process then branch and collect results
      const results = [];
      if (node.thenStatement) {
        if (ts.isBlock(node.thenStatement)) {
          const thenResult = processStatements(node.thenStatement.statements, context);
          results.push(...thenResult.results);
        } else {
          const result = processStatement(node.thenStatement, context);
          if (result) results.push(result);
        }
      }
      
      // Return all collected results as a special if-statement result
      return results.length > 0 ? { kind: 'if-results', results } : null;
      
    case ts.SyntaxKind.ExpressionStatement:
      infer(node.expression, context);
      return null;
      
    default:
      throw new Error(`Unsupported statement: ${ts.SyntaxKind[node.kind]}`);
  }
};

// Infer function body type
const inferFunctionBody = (bodyNode, context) => {
  if (ts.isBlock(bodyNode)) {
    const { results } = processStatements(bodyNode.statements, context);
    
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
    
    // No return statement = void function
    return hasReturn ? returnType : VOID_TYPE;
  } else {
    // Expression body
    return infer(bodyNode, context);
  }
};

// Process function declarations
const processProgram = (sourceCode) => {
  const sourceFile = ts.createSourceFile('temp.ts', sourceCode, ts.ScriptTarget.Latest, true);
  const context = {};
  const functions = [];
  const variables = [];
  
  for (const statement of sourceFile.statements) {
    if (ts.isFunctionDeclaration(statement)) {
      // Extract parameter types
      const paramTypes = [];
      for (const param of statement.parameters) {
        if (param.type) {
          paramTypes.push(convertTypeAnnotation(param.type));
        } else {
          throw new Error('Function parameters must have type annotations');
        }
      }
      
      // Build extended context with parameters
      const extendedContext = { ...context };
      for (let i = 0; i < statement.parameters.length; i++) {
        extendedContext[statement.parameters[i].name.text] = paramTypes[i];
      }
      
      // Infer or check return type
      let returnType;
      if (statement.type) {
        // Explicit return type
        returnType = convertTypeAnnotation(statement.type);
        const bodyType = inferFunctionBody(statement.body, extendedContext);
        if (!typesEqual(bodyType, returnType)) {
          throw new Error(`Function body returns ${formatType(bodyType)} but declared ${formatType(returnType)}`);
        }
      } else {
        // Infer return type
        returnType = inferFunctionBody(statement.body, extendedContext);
      }
      
      const functionType = createFunctionType(paramTypes, returnType);
      context[statement.name.text] = functionType;
      
      functions.push({
        name: statement.name.text,
        type: functionType
      });
    }
  }
  
  return { context, functions, variables };
};

// CLI functionality
const processFile = (filename) => {
  const fs = require('fs');
  
  try {
    const sourceCode = fs.readFileSync(filename, 'utf8');
    const result = processProgram(sourceCode);
    
    console.log(`\x1b[1m\x1b[36mType checking: ${filename}\x1b[0m`);
    console.log('Status: \x1b[32m✓ SUCCESS\x1b[0m');
    console.log();
    
    if (result.functions.length > 0) {
      console.log('\x1b[34mFunctions found:\x1b[0m');
      for (const func of result.functions) {
        console.log(`  \x1b[36m${func.name}\x1b[0m: \x1b[33m${formatType(func.type)}\x1b[0m`);
      }
      console.log();
    }
    
    console.log('\x1b[32m🎉 All types check out!\x1b[0m');
    
  } catch (error) {
    console.log(`\x1b[1m\x1b[36mType checking: ${filename}\x1b[0m`);
    console.log('Status: \x1b[31m✗ FAILED\x1b[0m');
    console.log();
    console.log('\x1b[31mType Error:\x1b[0m');
    console.log(`  ${error.message}`);
    console.log();
    console.log('\x1b[33m💡 Fix the type error and try again!\x1b[0m');
  }
};

// Main CLI
if (require.main === module) {
  const filename = process.argv[2];
  if (!filename) {
    console.log('Usage: node reduced.js <filename>');
    process.exit(1);
  }
  processFile(filename);
}

module.exports = {
  processProgram,
  formatType,
  NUMBER_TYPE,
  STRING_TYPE,
  BOOLEAN_TYPE,
  VOID_TYPE
};