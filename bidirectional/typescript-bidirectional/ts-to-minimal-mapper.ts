// Maps TypeScript AST to minimal typechecker expression format
import * as ts from 'typescript'
import { synth, check } from './minimal-typechecker'
import type { Type, Expr, Context } from './minimal-typechecker'

// Parse TypeScript type annotations to our Type format
function parseTypeAnnotation(typeNode: ts.TypeNode): Type {
  switch (typeNode.kind) {
    case ts.SyntaxKind.NumberKeyword:
      return { kind: 'number' }
    case ts.SyntaxKind.StringKeyword:
      return { kind: 'string' }
    case ts.SyntaxKind.FunctionType:
      const fnType = typeNode as ts.FunctionTypeNode
      if (fnType.parameters.length === 0) {
        throw new Error('Zero-parameter functions not supported in minimal typechecker')
      }
      const paramType = fnType.parameters[0].type!
      return {
        kind: 'function',
        arg: parseTypeAnnotation(paramType),
        returnType: parseTypeAnnotation(fnType.type)
      }
  }
  throw new Error(`Unsupported type annotation: ${ts.SyntaxKind[typeNode.kind]}`)
}

// Helper to create body expression from statements (with return)
function createBodyExpr(statements: readonly ts.Statement[], ctx: Context): Expr {
  const returnStmt = statements.find(ts.isReturnStatement)
  if (!returnStmt || !returnStmt.expression) {
    throw new Error('Function must have return statement')
  }

  // Get all non-return statements
  const nonReturnStmts = statements.filter(s => !ts.isReturnStatement(s))

  if (nonReturnStmts.length === 0) {
    // Just a return statement, no need for block
    return mapTsNodeToExpr(returnStmt.expression, ctx)
  } else {
    // Multiple statements - create a block
    return {
      kind: 'block',
      statements: nonReturnStmts.map(s => mapTsNodeToExpr(s, ctx)),
      return: mapTsNodeToExpr(returnStmt.expression, ctx)
    }
  }
}

// Map TypeScript AST nodes to minimal Expr format
function mapTsNodeToExpr(node: ts.Node, ctx: Context): Expr {
  switch (node.kind) {
    case ts.SyntaxKind.NumericLiteral:
      const numLit = node as ts.NumericLiteral
      return { kind: 'number', value: parseFloat(numLit.text) }

    case ts.SyntaxKind.StringLiteral:
      const strLit = node as ts.StringLiteral
      return { kind: 'string', value: strLit.text }

    case ts.SyntaxKind.Identifier:
      const ident = node as ts.Identifier
      return { kind: 'varLookup', name: ident.text }

    case ts.SyntaxKind.CallExpression:
      const call = node as ts.CallExpression
      return {
        kind: 'call',
        fn: mapTsNodeToExpr(call.expression, ctx),
        arg: mapTsNodeToExpr(call.arguments[0], ctx)
      }

    case ts.SyntaxKind.ArrowFunction:
    case ts.SyntaxKind.FunctionExpression:
      const func = node as ts.ArrowFunction | ts.FunctionExpression
      const param = func.parameters[0]
      const paramName = (param.name as ts.Identifier).text

      // Extract function body
      let bodyExpr: Expr
      if (ts.isBlock(func.body)) {
        bodyExpr = createBodyExpr(func.body.statements, ctx)
      } else {
        bodyExpr = mapTsNodeToExpr(func.body, ctx)
      }

      return {
        kind: 'function',
        param: paramName,
        body: bodyExpr
      }

    case ts.SyntaxKind.VariableStatement:
      const varStmt = node as ts.VariableStatement
      const decl = varStmt.declarationList.declarations[0]
      const varName = (decl.name as ts.Identifier).text

      // Check if initializer is a function expression/arrow with type annotation
      const init = decl.initializer!
      if ((ts.isArrowFunction(init) || ts.isFunctionExpression(init)) && init.type) {
        const param = init.parameters[0]
        const funcType: Type = {
          kind: 'function',
          arg: parseTypeAnnotation(param.type!),
          returnType: parseTypeAnnotation(init.type)
        }
        ctx.set(varName, funcType)
      }

      const valueExpr = mapTsNodeToExpr(decl.initializer!, ctx)

      // Parse type annotation if present
      const typeAnnotation = decl.type ? parseTypeAnnotation(decl.type) : undefined

      // Return a let statement (no body - will be used in a block)
      return {
        kind: 'let',
        name: varName,
        value: valueExpr,
        type: typeAnnotation
      }

    case ts.SyntaxKind.FunctionDeclaration:
      const funcDecl = node as ts.FunctionDeclaration
      const funcName = funcDecl.name!.text
      const funcParam = funcDecl.parameters[0]
      const funcParamName = (funcParam.name as ts.Identifier).text

      const funcBody = funcDecl.body as ts.Block
      const funcBodyExpr = createBodyExpr(funcBody.statements, ctx)

      const functionExpr: Expr = {
        kind: 'function',
        param: funcParamName,
        body: funcBodyExpr
      }

      // Store function type in context
      const funcType: Type = {
        kind: 'function',
        arg: parseTypeAnnotation(funcParam.type!),
        returnType: parseTypeAnnotation(funcDecl.type!)
      }
      ctx.set(funcName, funcType)

      return {
        kind: 'let',
        name: funcName,
        value: functionExpr
      }

    case ts.SyntaxKind.ExpressionStatement:
      const exprStmt = node as ts.ExpressionStatement
      return mapTsNodeToExpr(exprStmt.expression, ctx)
  }

  throw new Error(`Unsupported node kind: ${ts.SyntaxKind[node.kind]}`)
}

// Convert a sequence of statements into a block expression
function buildProgramExpr(statements: ts.Statement[], ctx: Context): Expr | null {
  if (statements.length === 0) {
    return null
  }

  if (statements.length === 1) {
    const expr = mapTsNodeToExpr(statements[0], ctx)
    // If it's a single let statement, wrap it in a block with varLookup return
    if (expr.kind === 'let') {
      return {
        kind: 'block',
        statements: [expr],
        return: { kind: 'varLookup', name: expr.name }
      }
    }
    return expr
  }

  // Multiple statements - separate let statements from the final expression
  const allExprs = statements.map(s => mapTsNodeToExpr(s, ctx))
  const letStmts = allExprs.slice(0, -1).filter(e => e.kind === 'let')
  const lastExpr = allExprs[allExprs.length - 1]

  // If the last expression is a let, we need to add it to statements and return a varLookup
  if (lastExpr.kind === 'let') {
    return {
      kind: 'block',
      statements: [...letStmts, lastExpr],
      return: { kind: 'varLookup', name: lastExpr.name }
    }
  }

  // Otherwise, use let statements with the last expression as the return
  return {
    kind: 'block',
    statements: letStmts,
    return: lastExpr
  }
}

// Main function to parse TypeScript code and typecheck with minimal typechecker
export function typecheckTypeScript(code: string): { expr: Expr, type: Type } {
  console.log(`\n=== TypeScript Code ===`)
  console.log(code)

  const sourceFile = ts.createSourceFile('test.ts', code, ts.ScriptTarget.Latest, true)
  const ctx: Context = new Map()

  // First pass: process function declarations and variable statements with function values
  const otherStmts: ts.Statement[] = []

  for (const stmt of sourceFile.statements) {
    if (ts.isFunctionDeclaration(stmt)) {
      // Handle function declarations
      const funcName = stmt.name!.text
      const funcParam = stmt.parameters[0]
      const funcParamName = (funcParam.name as ts.Identifier).text
      const funcBody = stmt.body as ts.Block
      const funcBodyExpr = createBodyExpr(funcBody.statements, ctx)

      const functionExpr: Expr = {
        kind: 'function',
        param: funcParamName,
        body: funcBodyExpr
      }

      const funcType: Type = {
        kind: 'function',
        arg: parseTypeAnnotation(funcParam.type!),
        returnType: parseTypeAnnotation(stmt.type!)
      }

      // Check the function against its type
      check(ctx, functionExpr, funcType)

      // Add to context
      ctx.set(funcName, funcType)
    } else if (ts.isVariableStatement(stmt)) {
      // Check if this is a variable with a function value
      const decl = stmt.declarationList.declarations[0]
      const init = decl.initializer

      if (init && (ts.isArrowFunction(init) || ts.isFunctionExpression(init)) && init.type) {
        // Handle function variables
        const varName = (decl.name as ts.Identifier).text
        const param = init.parameters[0]
        const paramName = (param.name as ts.Identifier).text

        let bodyExpr: Expr
        if (ts.isBlock(init.body)) {
          bodyExpr = createBodyExpr(init.body.statements, ctx)
        } else {
          bodyExpr = mapTsNodeToExpr(init.body, ctx)
        }

        const functionExpr: Expr = {
          kind: 'function',
          param: paramName,
          body: bodyExpr
        }

        const funcType: Type = {
          kind: 'function',
          arg: parseTypeAnnotation(param.type!),
          returnType: parseTypeAnnotation(init.type)
        }

        // Check the function against its type
        check(ctx, functionExpr, funcType)

        // Add to context
        ctx.set(varName, funcType)
      } else {
        // Non-function variable statement
        otherStmts.push(stmt)
      }
    } else {
      otherStmts.push(stmt)
    }
  }

  console.log(`\n=== Mapping to Minimal Expr ===`)
  const expr = buildProgramExpr(otherStmts, ctx)

  if (!expr) {
    throw new Error('No expression to type check (only function declarations)')
  }

  console.log(JSON.stringify(expr, null, 2))

  console.log(`\n=== Type Checking ===`)
  const type = synth(ctx, expr)
  console.log(`Type:`, type)

  return { expr, type }
}

// Demo - commented out, use demo.ts for examples
// console.log('=== TypeScript to Minimal Typechecker Mapper ===\n')
