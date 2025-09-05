import * as ts from 'typescript';

// Type system using TypeScript union types
type PrimitiveType = { kind: 'primitive'; name: 'number' | 'string' | 'boolean' | 'void' };
type FunctionType = { kind: 'function'; paramTypes: Type[]; returnType: Type };
type Type = PrimitiveType | FunctionType;

// Type constants
const NUMBER_TYPE: Type = { kind: 'primitive', name: 'number' };
const STRING_TYPE: Type = { kind: 'primitive', name: 'string' };
const BOOLEAN_TYPE: Type = { kind: 'primitive', name: 'boolean' };
const VOID_TYPE: Type = { kind: 'primitive', name: 'void' };

// Type equality - much simpler with TypeScript
const typesEqual = (a: Type, b: Type): boolean => {
  if (a.kind !== b.kind) return false;
  if (a.kind === 'primitive' && b.kind === 'primitive') return a.name === b.name;
  if (a.kind === 'function' && b.kind === 'function') {
    return a.paramTypes.length === b.paramTypes.length &&
           a.paramTypes.every((param, i) => typesEqual(param, b.paramTypes[i])) &&
           typesEqual(a.returnType, b.returnType);
  }
  return false;
};

// Convert TypeScript AST types to our types
const convertTypeAnnotation = (node: ts.TypeNode): Type => {
  switch (node.kind) {
    case ts.SyntaxKind.NumberKeyword: return NUMBER_TYPE;
    case ts.SyntaxKind.StringKeyword: return STRING_TYPE;
    case ts.SyntaxKind.BooleanKeyword: return BOOLEAN_TYPE;
    case ts.SyntaxKind.VoidKeyword: return VOID_TYPE;
    default: throw new Error(`Unsupported type: ${ts.SyntaxKind[node.kind]}`);
  }
};

// Format types for display
const formatType = (type: Type): string => {
  if (type.kind === 'primitive') return type.name;
  if (type.kind === 'function') {
    const params = type.paramTypes.map(formatType).join(', ');
    return `(${params}) => ${formatType(type.returnType)}`;
  }
  return 'unknown';
};

// Context is just a record of variable names to types
type Context = Record<string, Type>;

// Results from processing statements
type StatementResult = 
  | { kind: 'return'; type: Type }
  | { kind: 'if-results'; results: StatementResult[] };

// Type inference
const infer = (node: ts.Node, context: Context): Type => {
  switch (node.kind) {
    case ts.SyntaxKind.NumericLiteral: return NUMBER_TYPE;
    case ts.SyntaxKind.StringLiteral: return STRING_TYPE;
    case ts.SyntaxKind.TrueKeyword:
    case ts.SyntaxKind.FalseKeyword: return BOOLEAN_TYPE;
    
    case ts.SyntaxKind.Identifier: {
      const name = (node as ts.Identifier).text;
      const type = context[name];
      if (!type) throw new Error(`Variable '${name}' is not defined`);
      return type;
    }
    
    case ts.SyntaxKind.BinaryExpression: {
      const binary = node as ts.BinaryExpression;
      const left = infer(binary.left, context);
      const right = infer(binary.right, context);
      
      switch (binary.operatorToken.kind) {
        case ts.SyntaxKind.PlusToken:
          // Arithmetic or concatenation
          if (typesEqual(left, NUMBER_TYPE) && typesEqual(right, NUMBER_TYPE)) return NUMBER_TYPE;
          if (typesEqual(left, STRING_TYPE) && typesEqual(right, STRING_TYPE)) return STRING_TYPE;
          if (typesEqual(left, STRING_TYPE) && typesEqual(right, NUMBER_TYPE)) return STRING_TYPE;
          if (typesEqual(left, NUMBER_TYPE) && typesEqual(right, STRING_TYPE)) return STRING_TYPE;
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
          throw new Error(`Unsupported operator: ${ts.SyntaxKind[binary.operatorToken.kind]}`);
      }
    }
    
    case ts.SyntaxKind.CallExpression: {
      const call = node as ts.CallExpression;
      const funcType = infer(call.expression, context);
      
      if (funcType.kind !== 'function') throw new Error('Cannot call non-function');
      if (call.arguments.length !== funcType.paramTypes.length) {
        throw new Error(`Expected ${funcType.paramTypes.length} arguments, got ${call.arguments.length}`);
      }
      
      // Check argument types
      call.arguments.forEach((arg, i) => {
        const argType = infer(arg, context);
        if (!typesEqual(argType, funcType.paramTypes[i])) {
          throw new Error(`Argument ${i} expects ${formatType(funcType.paramTypes[i])}, got ${formatType(argType)}`);
        }
      });
      
      return funcType.returnType;
    }
    
    default:
      throw new Error(`Unsupported expression: ${ts.SyntaxKind[node.kind]}`);
  }
};

// Process statements
const processStatements = (statements: readonly ts.Statement[], context: Context) => {
  const newContext = { ...context };
  const results: StatementResult[] = [];
  
  for (const stmt of statements) {
    const result = processStatement(stmt, newContext);
    if (result) {
      if (result.kind === 'if-results') {
        results.push(...result.results);
      } else {
        results.push(result);
      }
    }
  }
  
  return { context: newContext, results };
};

// Process individual statement
const processStatement = (node: ts.Statement, context: Context): StatementResult | null => {
  switch (node.kind) {
    case ts.SyntaxKind.VariableStatement: {
      const varStmt = node as ts.VariableStatement;
      for (const decl of varStmt.declarationList.declarations) {
        if (decl.initializer) {
          const type = infer(decl.initializer, context);
          context[(decl.name as ts.Identifier).text] = type;
        }
      }
      return null;
    }
    
    case ts.SyntaxKind.ReturnStatement: {
      const returnStmt = node as ts.ReturnStatement;
      const type = returnStmt.expression ? infer(returnStmt.expression, context) : VOID_TYPE;
      return { kind: 'return', type };
    }
    
    case ts.SyntaxKind.IfStatement: {
      const ifStmt = node as ts.IfStatement;
      
      // Process condition
      infer(ifStmt.expression, context);
      
      // Process then branch
      const results: StatementResult[] = [];
      if (ifStmt.thenStatement) {
        if (ts.isBlock(ifStmt.thenStatement)) {
          const thenResult = processStatements(ifStmt.thenStatement.statements, context);
          results.push(...thenResult.results);
        } else {
          const result = processStatement(ifStmt.thenStatement, context);
          if (result) results.push(result);
        }
      }
      
      return results.length > 0 ? { kind: 'if-results', results } : null;
    }
    
    case ts.SyntaxKind.ExpressionStatement:
      infer((node as ts.ExpressionStatement).expression, context);
      return null;
      
    default:
      throw new Error(`Unsupported statement: ${ts.SyntaxKind[node.kind]}`);
  }
};

// Infer function body type with return consistency checking
const inferFunctionBody = (body: ts.ConciseBody, context: Context): Type => {
  if (ts.isBlock(body)) {
    const { results } = processStatements(body.statements, context);
    
    // Validate all return statements have the same type
    let returnType: Type | null = null;
    let hasReturn = false;
    
    for (const result of results) {
      if (result.kind === 'return') {
        if (!hasReturn) {
          returnType = result.type;
          hasReturn = true;
        } else if (!typesEqual(result.type, returnType!)) {
          throw new Error(`All return statements must have the same type. Expected ${formatType(returnType!)}, got ${formatType(result.type)}`);
        }
      }
    }
    
    return hasReturn ? returnType! : VOID_TYPE;
  } else {
    // Expression body
    return infer(body, context);
  }
};

// Main program processor
const processProgram = (sourceCode: string) => {
  const sourceFile = ts.createSourceFile('temp.ts', sourceCode, ts.ScriptTarget.Latest, true);
  const context: Context = {};
  const functions: { name: string; type: FunctionType }[] = [];
  
  for (const stmt of sourceFile.statements) {
    if (ts.isFunctionDeclaration(stmt) && stmt.name) {
      // Extract parameter types
      const paramTypes = stmt.parameters.map(param => {
        if (!param.type) throw new Error('Function parameters must have type annotations');
        return convertTypeAnnotation(param.type);
      });
      
      // Build context with parameters
      const extendedContext = { ...context };
      stmt.parameters.forEach((param, i) => {
        extendedContext[(param.name as ts.Identifier).text] = paramTypes[i];
      });
      
      // Infer or check return type
      let returnType: Type;
      if (stmt.type) {
        returnType = convertTypeAnnotation(stmt.type);
        const bodyType = inferFunctionBody(stmt.body!, extendedContext);
        if (!typesEqual(bodyType, returnType)) {
          throw new Error(`Function body returns ${formatType(bodyType)} but declared ${formatType(returnType)}`);
        }
      } else {
        returnType = inferFunctionBody(stmt.body!, extendedContext);
      }
      
      const functionType: FunctionType = { kind: 'function', paramTypes, returnType };
      context[stmt.name.text] = functionType;
      functions.push({ name: stmt.name.text, type: functionType });
    }
  }
  
  return { context, functions };
};

// CLI functionality
const processFile = (filename: string) => {
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
    console.log('Usage: npx ts-node reduced.ts <filename>');
    process.exit(1);
  }
  processFile(filename);
}

export { processProgram, formatType, NUMBER_TYPE, STRING_TYPE, BOOLEAN_TYPE, VOID_TYPE };