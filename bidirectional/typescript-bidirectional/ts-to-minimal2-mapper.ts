// Maps TypeScript AST to extended minimal typechecker expression format
import ts from 'typescript'
import { Type, Expr, Context, synth, check, typeToString } from './minimal2-typechecker.js'

// Parse TypeScript type annotations to our Type format
function parseTypeAnnotation(typeNode: ts.TypeNode | undefined): Type {
  if (!typeNode) {
    return { kind: 'void' }
  }

  switch (typeNode.kind) {
    case ts.SyntaxKind.NumberKeyword:
      return { kind: 'number' }
    case ts.SyntaxKind.StringKeyword:
      return { kind: 'string' }
    case ts.SyntaxKind.BooleanKeyword:
      return { kind: 'boolean' }
    case ts.SyntaxKind.VoidKeyword:
      return { kind: 'void' }
    case ts.SyntaxKind.FunctionType:
      const fnType = typeNode as ts.FunctionTypeNode
      const params = fnType.parameters.map(p => parseTypeAnnotation(p.type))
      return {
        kind: 'function',
        params,
        to: parseTypeAnnotation(fnType.type)
      }
  }
  throw new Error(`Unsupported type annotation: ${ts.SyntaxKind[typeNode.kind]}`)
}

// Map binary operator tokens
function mapBinaryOp(op: ts.BinaryOperator): '+' | '*' | '>' | '<' | '>=' | '<=' | '===' | '!==' | '-' | '/' {
  switch (op) {
    case ts.SyntaxKind.PlusToken: return '+'
    case ts.SyntaxKind.MinusToken: return '-'
    case ts.SyntaxKind.AsteriskToken: return '*'
    case ts.SyntaxKind.SlashToken: return '/'
    case ts.SyntaxKind.GreaterThanToken: return '>'
    case ts.SyntaxKind.LessThanToken: return '<'
    case ts.SyntaxKind.GreaterThanEqualsToken: return '>='
    case ts.SyntaxKind.LessThanEqualsToken: return '<='
    case ts.SyntaxKind.EqualsEqualsEqualsToken: return '==='
    case ts.SyntaxKind.ExclamationEqualsEqualsToken: return '!=='
    default:
      throw new Error(`Unsupported binary operator: ${ts.SyntaxKind[op]}`)
  }
}

// Map TypeScript AST nodes to minimal2 Expr format
function mapTsNodeToExpr(node: ts.Node, ctx: Map<string, Type>): Expr {
  switch (node.kind) {
    case ts.SyntaxKind.NumericLiteral:
      const numLit = node as ts.NumericLiteral
      return { kind: 'number', value: parseFloat(numLit.text) }

    case ts.SyntaxKind.StringLiteral:
      const strLit = node as ts.StringLiteral
      return { kind: 'string', value: strLit.text }

    case ts.SyntaxKind.TrueKeyword:
      return { kind: 'boolean', value: true }

    case ts.SyntaxKind.FalseKeyword:
      return { kind: 'boolean', value: false }

    case ts.SyntaxKind.Identifier:
      const ident = node as ts.Identifier
      return { kind: 'var', name: ident.text }

    case ts.SyntaxKind.CallExpression:
      const call = node as ts.CallExpression
      return {
        kind: 'app',
        fn: mapTsNodeToExpr(call.expression, ctx),
        args: call.arguments.map(arg => mapTsNodeToExpr(arg, ctx))
      }

    case ts.SyntaxKind.BinaryExpression:
      const binExpr = node as ts.BinaryExpression
      return {
        kind: 'binop',
        op: mapBinaryOp(binExpr.operatorToken.kind),
        left: mapTsNodeToExpr(binExpr.left, ctx),
        right: mapTsNodeToExpr(binExpr.right, ctx)
      }

    case ts.SyntaxKind.IfStatement:
      const ifStmt = node as ts.IfStatement
      return {
        kind: 'if',
        condition: mapTsNodeToExpr(ifStmt.expression, ctx),
        then: mapTsNodeToExpr(ifStmt.thenStatement, ctx),
        else: ifStmt.elseStatement
          ? mapTsNodeToExpr(ifStmt.elseStatement, ctx)
          : { kind: 'sequence', exprs: [] } // void
      }

    case ts.SyntaxKind.Block:
      const block = node as ts.Block
      return mapBlockToExpr(block, ctx)

    case ts.SyntaxKind.ReturnStatement:
      const returnStmt = node as ts.ReturnStatement
      if (!returnStmt.expression) {
        return { kind: 'sequence', exprs: [] } // void return
      }
      return mapTsNodeToExpr(returnStmt.expression, ctx)

    case ts.SyntaxKind.ArrowFunction:
    case ts.SyntaxKind.FunctionExpression:
      const func = node as ts.ArrowFunction | ts.FunctionExpression
      const params = func.parameters.map(p => (p.name as ts.Identifier).text)

      let bodyExpr: Expr
      if (ts.isBlock(func.body)) {
        bodyExpr = mapBlockToExpr(func.body, ctx)
      } else {
        bodyExpr = mapTsNodeToExpr(func.body, ctx)
      }

      return {
        kind: 'lambda',
        params,
        body: bodyExpr
      }

    case ts.SyntaxKind.VariableStatement:
      const varStmt = node as ts.VariableStatement
      const decl = varStmt.declarationList.declarations[0]
      const varName = (decl.name as ts.Identifier).text

      // Check if initializer is a function expression/arrow with type annotation
      const init = decl.initializer!
      if ((ts.isArrowFunction(init) || ts.isFunctionExpression(init)) && (init.type || init.parameters.some(p => p.type))) {
        const params = init.parameters.map(p => parseTypeAnnotation(p.type))
        const returnType = parseTypeAnnotation(init.type)
        const funcType: Type = {
          kind: 'function',
          params,
          to: returnType
        }
        ctx.set(varName, funcType)
      }

      const valueExpr = mapTsNodeToExpr(decl.initializer!, ctx)

      return {
        kind: 'let',
        name: varName,
        value: valueExpr,
        body: { kind: 'var', name: varName } // placeholder
      }

    case ts.SyntaxKind.FunctionDeclaration:
      const funcDecl = node as ts.FunctionDeclaration
      const funcName = funcDecl.name!.text
      const funcParams = funcDecl.parameters.map(p => (p.name as ts.Identifier).text)

      // Store function type in context
      const funcType: Type = {
        kind: 'function',
        params: funcDecl.parameters.map(p => parseTypeAnnotation(p.type)),
        to: parseTypeAnnotation(funcDecl.type)
      }
      ctx.set(funcName, funcType)

      // Create context for function body with parameters
      const bodyCtx = new Map(ctx)
      for (let i = 0; i < funcParams.length; i++) {
        bodyCtx.set(funcParams[i], funcType.params[i])
      }

      const lambdaExpr: Expr = {
        kind: 'lambda',
        params: funcParams,
        body: mapBlockToExpr(funcDecl.body!, bodyCtx)
      }

      return {
        kind: 'let',
        name: funcName,
        value: lambdaExpr,
        body: { kind: 'var', name: funcName } // placeholder
      }

    case ts.SyntaxKind.ExpressionStatement:
      const exprStmt = node as ts.ExpressionStatement
      return mapTsNodeToExpr(exprStmt.expression, ctx)
  }

  throw new Error(`Unsupported node kind: ${ts.SyntaxKind[node.kind]}`)
}

// Map a block statement to an expression, handling multiple statements
function mapBlockToExpr(block: ts.Block, ctx: Map<string, Type>): Expr {
  const statements = Array.from(block.statements)

  if (statements.length === 0) {
    return { kind: 'sequence', exprs: [] }
  }

  // Process statements sequentially, building up context and nested lets
  return mapStatementsToExpr(statements, ctx)
}

function mapStatementsToExpr(statements: ts.Statement[], ctx: Map<string, Type>): Expr {
  if (statements.length === 0) {
    return { kind: 'sequence', exprs: [] }
  }

  const [first, ...rest] = statements

  // Handle variable declarations
  if (ts.isVariableStatement(first)) {
    const decl = first.declarationList.declarations[0]
    const varName = (decl.name as ts.Identifier).text
    const valueExpr = mapTsNodeToExpr(decl.initializer!, ctx)

    // Infer type from initializer and add to context for subsequent statements
    const localCtx = new Map(ctx)
    const valueType = synth(localCtx, valueExpr)
    localCtx.set(varName, valueType)

    // Build the body with the updated context
    const bodyExpr = rest.length > 0
      ? mapStatementsToExpr(rest, localCtx)
      : { kind: 'sequence', exprs: [] }

    return {
      kind: 'let',
      name: varName,
      value: valueExpr,
      body: bodyExpr
    }
  }

  // Handle return statements
  if (ts.isReturnStatement(first)) {
    return first.expression
      ? mapTsNodeToExpr(first.expression, ctx)
      : { kind: 'sequence', exprs: [] }
  }

  // Handle if statements specially - they might have early returns
  if (ts.isIfStatement(first)) {
    const ifExpr = mapTsNodeToExpr(first, ctx)

    // If there are more statements after the if, we need to handle it carefully
    if (rest.length > 0) {
      // Check if the if statement has returns in both branches
      const hasReturnInThen = hasReturnStatement(first.thenStatement)
      const hasReturnInElse = first.elseStatement ? hasReturnStatement(first.elseStatement) : false

      if (hasReturnInThen && hasReturnInElse) {
        // Both branches return, so the if is the final result
        return ifExpr
      } else if (!hasReturnInThen && !hasReturnInElse) {
        // Neither branch returns, continue with rest
        const restExpr = mapStatementsToExpr(rest, ctx)
        return {
          kind: 'sequence',
          exprs: [ifExpr, restExpr]
        }
      } else {
        // One branch returns, one doesn't - need to transform
        // The if should produce the final value, with the non-returning branch continuing
        const restExpr = mapStatementsToExpr(rest, ctx)

        if (hasReturnInThen && !hasReturnInElse) {
          // Transform: if (cond) { return X } else { ... }; Y
          // Into: if (cond) X else { ...; Y }
          return {
            kind: 'if',
            condition: (ifExpr as any).condition,
            then: (ifExpr as any).then,
            else: (ifExpr as any).else.kind === 'sequence' && (ifExpr as any).else.exprs.length === 0
              ? restExpr
              : { kind: 'sequence', exprs: [(ifExpr as any).else, restExpr] }
          }
        } else {
          // Transform: if (cond) { ... } else { return X }; Y
          // Into: if (cond) { ...; Y } else X
          return {
            kind: 'if',
            condition: (ifExpr as any).condition,
            then: (ifExpr as any).then.kind === 'sequence' && (ifExpr as any).then.exprs.length === 0
              ? restExpr
              : { kind: 'sequence', exprs: [(ifExpr as any).then, restExpr] },
            else: (ifExpr as any).else
          }
        }
      }
    }

    return ifExpr
  }

  // Handle other statements (like expression statements)
  const firstExpr = mapTsNodeToExpr(first, ctx)

  if (rest.length === 0) {
    return firstExpr
  }

  // If we have more statements, create a sequence
  const restExpr = mapStatementsToExpr(rest, ctx)

  // If the rest is a single expression, create a proper sequence
  if (restExpr.kind === 'sequence' && restExpr.exprs.length === 0) {
    return firstExpr
  }

  return {
    kind: 'sequence',
    exprs: [firstExpr, restExpr]
  }
}

// Helper to check if a statement contains a return
function hasReturnStatement(stmt: ts.Statement): boolean {
  if (ts.isReturnStatement(stmt)) {
    return true
  }
  if (ts.isBlock(stmt)) {
    return stmt.statements.some(s => hasReturnStatement(s))
  }
  return false
}

// Convert a sequence of statements into nested let expressions
function buildProgramExpr(statements: ts.Statement[], ctx: Map<string, Type>): Expr {
  if (statements.length === 0) {
    throw new Error('Empty program')
  }

  if (statements.length === 1) {
    return mapTsNodeToExpr(statements[0], ctx)
  }

  const [first, ...rest] = statements
  const firstExpr = mapTsNodeToExpr(first, ctx)

  if (firstExpr.kind === 'let') {
    return {
      kind: 'let',
      name: firstExpr.name,
      value: firstExpr.value,
      body: buildProgramExpr(rest, ctx)
    }
  } else {
    return buildProgramExpr(rest, ctx)
  }
}

// Main function to parse TypeScript code and typecheck with minimal2 typechecker
export function typecheckTypeScript(code: string): { expr: Expr, type: Type } {
  console.log(`\n=== TypeScript Code ===`)
  console.log(code)

  const sourceFile = ts.createSourceFile('test.ts', code, ts.ScriptTarget.Latest, true)
  const ctx = new Map<string, Type>()

  console.log(`\n=== Mapping to Minimal2 Expr ===`)
  const expr = buildProgramExpr(Array.from(sourceFile.statements), ctx)
  console.log(JSON.stringify(expr, null, 2))

  console.log(`\n=== Type Checking ===`)
  const type = synth(ctx, expr)
  console.log(`Type: ${typeToString(type)}`)

  return { expr, type }
}

// Demo
console.log('=== TypeScript to Minimal2 Typechecker Mapper ===\n')

// Example 1: Binary operations
typecheckTypeScript('1 + 2')

// Example 2: Multiple parameters
typecheckTypeScript(`
function add(x: number, y: number): number {
  return x + y
}
add(10, 20)
`.trim())

// Example 3: If expression
typecheckTypeScript(`
function max(a: number, b: number): number {
  if (a > b) {
    return a
  } else {
    return b
  }
}
max(5, 10)
`.trim())

// Example 4: String concatenation
typecheckTypeScript(`
function greet(name: string): string {
  return "Hello, " + name
}
greet("World")
`.trim())

// Example 5: Void function
typecheckTypeScript(`
function log(message: string): void {

}
log("test")
`.trim())

// Example 6: Complex shipping calculation
console.log('\n\n=== SHIPPING CALCULATION EXAMPLE ===')
typecheckTypeScript(`
function log(message: string): void {

}

function calculateShipping(weight: number, distance: number): number {
  let baseRate = 5.00
  let weightFee = weight * 0.50
  let distanceFee = distance * 0.25
  let freeShippingThreshold = 75.0

  let total = baseRate + weightFee + distanceFee

  if (total > freeShippingThreshold) {
    log("Free shipping applied! Saved: $" + total)
    return 0.0
  }

  return total
}

calculateShipping(10, 100)
`.trim())
