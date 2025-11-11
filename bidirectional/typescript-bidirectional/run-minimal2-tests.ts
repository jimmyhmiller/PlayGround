// Test runner for minimal2 typechecker
import ts from 'typescript'
import * as fs from 'fs'
import * as path from 'path'

type Type =
  | { kind: 'number' }
  | { kind: 'string' }
  | { kind: 'boolean' }
  | { kind: 'void' }
  | { kind: 'function', params: Type[], to: Type }

type Expr =
  | { kind: 'number', value: number }
  | { kind: 'string', value: string }
  | { kind: 'boolean', value: boolean }
  | { kind: 'var', name: string }
  | { kind: 'lambda', params: string[], body: Expr }
  | { kind: 'app', fn: Expr, args: Expr[] }
  | { kind: 'let', name: string, value: Expr, body: Expr }
  | { kind: 'if', condition: Expr, then: Expr, else: Expr }
  | { kind: 'binop', op: '+' | '*' | '>' | '<' | '>=' | '<=' | '===' | '!==' | '-' | '/', left: Expr, right: Expr }
  | { kind: 'sequence', exprs: Expr[] }

type Context = Map<string, Type>

function synth(ctx: Context, expr: Expr): Type {
  switch (expr.kind) {
    case 'number': return { kind: 'number' }
    case 'string': return { kind: 'string' }
    case 'boolean': return { kind: 'boolean' }
    case 'var':
      const type = ctx.get(expr.name)
      if (!type) throw new Error(`Unbound variable: ${expr.name}`)
      return type
    case 'app':
      const fnType = synth(ctx, expr.fn)
      if (fnType.kind !== 'function') throw new Error('Cannot apply non-function')
      if (fnType.params.length !== expr.args.length) {
        throw new Error(`Expected ${fnType.params.length} arguments, got ${expr.args.length}`)
      }
      for (let i = 0; i < expr.args.length; i++) {
        check(ctx, expr.args[i], fnType.params[i])
      }
      return fnType.to
    case 'lambda':
      throw new Error('Cannot synthesize type for lambda without annotation')
    case 'let':
      const expectedType = ctx.get(expr.name)
      let valueType: Type
      if (expectedType && expr.value.kind === 'lambda') {
        check(ctx, expr.value, expectedType)
        valueType = expectedType
      } else {
        valueType = synth(ctx, expr.value)
      }
      const newCtx = new Map(ctx)
      newCtx.set(expr.name, valueType)
      return synth(newCtx, expr.body)
    case 'if':
      check(ctx, expr.condition, { kind: 'boolean' })
      const thenType = synth(ctx, expr.then)
      check(ctx, expr.else, thenType)
      return thenType
    case 'binop':
      switch (expr.op) {
        case '+':
          const leftType = synth(ctx, expr.left)
          const rightType = synth(ctx, expr.right)
          if (leftType.kind === 'number' && rightType.kind === 'number') return { kind: 'number' }
          if (leftType.kind === 'string' && rightType.kind === 'string') return { kind: 'string' }
          if ((leftType.kind === 'string' && rightType.kind === 'number') ||
              (leftType.kind === 'number' && rightType.kind === 'string')) return { kind: 'string' }
          throw new Error(`Cannot add ${leftType.kind} and ${rightType.kind}`)
        case '-':
        case '*':
        case '/':
          check(ctx, expr.left, { kind: 'number' })
          check(ctx, expr.right, { kind: 'number' })
          return { kind: 'number' }
        case '>':
        case '<':
        case '>=':
        case '<=':
          check(ctx, expr.left, { kind: 'number' })
          check(ctx, expr.right, { kind: 'number' })
          return { kind: 'boolean' }
        case '===':
        case '!==':
          const leftEqType = synth(ctx, expr.left)
          check(ctx, expr.right, leftEqType)
          return { kind: 'boolean' }
      }
    case 'sequence':
      if (expr.exprs.length === 0) return { kind: 'void' }
      for (let i = 0; i < expr.exprs.length - 1; i++) {
        synth(ctx, expr.exprs[i])
      }
      return synth(ctx, expr.exprs[expr.exprs.length - 1])
  }
}

function check(ctx: Context, expr: Expr, expected: Type): void {
  switch (expr.kind) {
    case 'lambda':
      if (expected.kind !== 'function') throw new Error('Lambda must have function type')
      if (expected.params.length !== expr.params.length) {
        throw new Error(`Expected ${expected.params.length} parameters, got ${expr.params.length}`)
      }
      const newCtx = new Map(ctx)
      for (let i = 0; i < expr.params.length; i++) {
        newCtx.set(expr.params[i], expected.params[i])
      }
      check(newCtx, expr.body, expected.to)
      break
    default:
      const actual = synth(ctx, expr)
      if (!typesEqual(actual, expected)) {
        throw new Error(`Type mismatch: expected ${typeToString(expected)}, got ${typeToString(actual)}`)
      }
  }
}

function typesEqual(a: Type, b: Type): boolean {
  if (a.kind !== b.kind) return false
  if (a.kind === 'function' && b.kind === 'function') {
    if (a.params.length !== b.params.length) return false
    for (let i = 0; i < a.params.length; i++) {
      if (!typesEqual(a.params[i], b.params[i])) return false
    }
    return typesEqual(a.to, b.to)
  }
  return true
}

function typeToString(type: Type): string {
  switch (type.kind) {
    case 'number': return 'number'
    case 'string': return 'string'
    case 'boolean': return 'boolean'
    case 'void': return 'void'
    case 'function':
      const params = type.params.map(typeToString).join(', ')
      return `(${params}) -> ${typeToString(type.to)}`
  }
}

function parseTypeAnnotation(typeNode: ts.TypeNode | undefined): Type {
  if (!typeNode) return { kind: 'void' }
  switch (typeNode.kind) {
    case ts.SyntaxKind.NumberKeyword: return { kind: 'number' }
    case ts.SyntaxKind.StringKeyword: return { kind: 'string' }
    case ts.SyntaxKind.BooleanKeyword: return { kind: 'boolean' }
    case ts.SyntaxKind.VoidKeyword: return { kind: 'void' }
    case ts.SyntaxKind.FunctionType:
      const fnType = typeNode as ts.FunctionTypeNode
      return {
        kind: 'function',
        params: fnType.parameters.map(p => parseTypeAnnotation(p.type)),
        to: parseTypeAnnotation(fnType.type)
      }
  }
  throw new Error(`Unsupported type annotation: ${ts.SyntaxKind[typeNode.kind]}`)
}

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
    default: throw new Error(`Unsupported binary operator: ${ts.SyntaxKind[op]}`)
  }
}

function mapTsNodeToExpr(node: ts.Node, ctx: Map<string, Type>): Expr {
  switch (node.kind) {
    case ts.SyntaxKind.NumericLiteral:
      return { kind: 'number', value: parseFloat((node as ts.NumericLiteral).text) }
    case ts.SyntaxKind.StringLiteral:
      return { kind: 'string', value: (node as ts.StringLiteral).text }
    case ts.SyntaxKind.TrueKeyword: return { kind: 'boolean', value: true }
    case ts.SyntaxKind.FalseKeyword: return { kind: 'boolean', value: false }
    case ts.SyntaxKind.Identifier:
      return { kind: 'var', name: (node as ts.Identifier).text }
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
        else: ifStmt.elseStatement ? mapTsNodeToExpr(ifStmt.elseStatement, ctx) : { kind: 'sequence', exprs: [] }
      }
    case ts.SyntaxKind.Block:
      return mapBlockToExpr(node as ts.Block, ctx)
    case ts.SyntaxKind.ReturnStatement:
      const returnStmt = node as ts.ReturnStatement
      return returnStmt.expression ? mapTsNodeToExpr(returnStmt.expression, ctx) : { kind: 'sequence', exprs: [] }
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
      return { kind: 'lambda', params, body: bodyExpr }
    case ts.SyntaxKind.VariableStatement:
      const varStmt = node as ts.VariableStatement
      const decl = varStmt.declarationList.declarations[0]
      const varName = (decl.name as ts.Identifier).text
      const init = decl.initializer!
      if ((ts.isArrowFunction(init) || ts.isFunctionExpression(init)) && (init.type || init.parameters.some(p => p.type))) {
        const params = init.parameters.map(p => parseTypeAnnotation(p.type))
        const returnType = parseTypeAnnotation(init.type)
        ctx.set(varName, { kind: 'function', params, to: returnType })
      }
      const valueExpr = mapTsNodeToExpr(decl.initializer!, ctx)
      return { kind: 'let', name: varName, value: valueExpr, body: { kind: 'var', name: varName } }
    case ts.SyntaxKind.FunctionDeclaration:
      const funcDecl = node as ts.FunctionDeclaration
      const funcName = funcDecl.name!.text
      const funcParams = funcDecl.parameters.map(p => (p.name as ts.Identifier).text)
      const funcType: Type = {
        kind: 'function',
        params: funcDecl.parameters.map(p => parseTypeAnnotation(p.type)),
        to: parseTypeAnnotation(funcDecl.type)
      }
      ctx.set(funcName, funcType)
      const bodyCtx = new Map(ctx)
      for (let i = 0; i < funcParams.length; i++) {
        bodyCtx.set(funcParams[i], funcType.params[i])
      }
      const lambdaExpr: Expr = { kind: 'lambda', params: funcParams, body: mapBlockToExpr(funcDecl.body!, bodyCtx) }
      return { kind: 'let', name: funcName, value: lambdaExpr, body: { kind: 'var', name: funcName } }
    case ts.SyntaxKind.ExpressionStatement:
      return mapTsNodeToExpr((node as ts.ExpressionStatement).expression, ctx)
  }
  throw new Error(`Unsupported node kind: ${ts.SyntaxKind[node.kind]}`)
}

function mapBlockToExpr(block: ts.Block, ctx: Map<string, Type>): Expr {
  const statements = Array.from(block.statements)
  if (statements.length === 0) return { kind: 'sequence', exprs: [] }
  return mapStatementsToExpr(statements, ctx)
}

function mapStatementsToExpr(statements: ts.Statement[], ctx: Map<string, Type>): Expr {
  if (statements.length === 0) return { kind: 'sequence', exprs: [] }
  const [first, ...rest] = statements
  if (ts.isVariableStatement(first)) {
    const decl = first.declarationList.declarations[0]
    const varName = (decl.name as ts.Identifier).text
    const valueExpr = mapTsNodeToExpr(decl.initializer!, ctx)
    const localCtx = new Map(ctx)
    const valueType = synth(localCtx, valueExpr)
    localCtx.set(varName, valueType)
    const bodyExpr = rest.length > 0 ? mapStatementsToExpr(rest, localCtx) : { kind: 'sequence', exprs: [] }
    return { kind: 'let', name: varName, value: valueExpr, body: bodyExpr }
  }
  if (ts.isReturnStatement(first)) {
    return first.expression ? mapTsNodeToExpr(first.expression, ctx) : { kind: 'sequence', exprs: [] }
  }
  if (ts.isIfStatement(first)) {
    const ifExpr = mapTsNodeToExpr(first, ctx)
    if (rest.length > 0) {
      const hasReturnInThen = hasReturnStatement(first.thenStatement)
      const hasReturnInElse = first.elseStatement ? hasReturnStatement(first.elseStatement) : false
      if (hasReturnInThen && hasReturnInElse) return ifExpr
      if (!hasReturnInThen && !hasReturnInElse) {
        const restExpr = mapStatementsToExpr(rest, ctx)
        return { kind: 'sequence', exprs: [ifExpr, restExpr] }
      }
      const restExpr = mapStatementsToExpr(rest, ctx)
      if (hasReturnInThen && !hasReturnInElse) {
        return {
          kind: 'if',
          condition: (ifExpr as any).condition,
          then: (ifExpr as any).then,
          else: (ifExpr as any).else.kind === 'sequence' && (ifExpr as any).else.exprs.length === 0
            ? restExpr : { kind: 'sequence', exprs: [(ifExpr as any).else, restExpr] }
        }
      } else {
        return {
          kind: 'if',
          condition: (ifExpr as any).condition,
          then: (ifExpr as any).then.kind === 'sequence' && (ifExpr as any).then.exprs.length === 0
            ? restExpr : { kind: 'sequence', exprs: [(ifExpr as any).then, restExpr] },
          else: (ifExpr as any).else
        }
      }
    }
    return ifExpr
  }
  const firstExpr = mapTsNodeToExpr(first, ctx)
  if (rest.length === 0) return firstExpr
  const restExpr = mapStatementsToExpr(rest, ctx)
  if (restExpr.kind === 'sequence' && restExpr.exprs.length === 0) return firstExpr
  return { kind: 'sequence', exprs: [firstExpr, restExpr] }
}

function hasReturnStatement(stmt: ts.Statement): boolean {
  if (ts.isReturnStatement(stmt)) return true
  if (ts.isBlock(stmt)) return stmt.statements.some(s => hasReturnStatement(s))
  return false
}

function buildProgramExpr(statements: ts.Statement[], ctx: Map<string, Type>): Expr {
  if (statements.length === 0) throw new Error('Empty program')
  if (statements.length === 1) return mapTsNodeToExpr(statements[0], ctx)
  const [first, ...rest] = statements
  const firstExpr = mapTsNodeToExpr(first, ctx)
  if (firstExpr.kind === 'let') {
    return { kind: 'let', name: firstExpr.name, value: firstExpr.value, body: buildProgramExpr(rest, ctx) }
  } else {
    return buildProgramExpr(rest, ctx)
  }
}

function testFile(filePath: string): { pass: boolean, error?: string } {
  try {
    const code = fs.readFileSync(filePath, 'utf-8')
    const sourceFile = ts.createSourceFile('test.ts', code, ts.ScriptTarget.Latest, true)
    const ctx = new Map<string, Type>()
    const expr = buildProgramExpr(Array.from(sourceFile.statements), ctx)
    synth(ctx, expr)
    return { pass: true }
  } catch (e) {
    return { pass: false, error: (e as Error).message }
  }
}

// Run all tests
const testsDir = './tests'
const testFiles = fs.readdirSync(testsDir).filter(f => f.endsWith('.ts'))

console.log('=== Running Minimal2 Tests ===\n')

let passCount = 0
let failCount = 0

for (const file of testFiles) {
  const filePath = path.join(testsDir, file)
  const result = testFile(filePath)

  const status = result.pass ? '✓ PASS' : '✗ FAIL'
  console.log(`${status} - ${file}`)

  if (!result.pass) {
    console.log(`  Error: ${result.error}`)
    failCount++
  } else {
    passCount++
  }
}

console.log(`\n=== Summary ===`)
console.log(`Passed: ${passCount}/${testFiles.length}`)
console.log(`Failed: ${failCount}/${testFiles.length}`)
