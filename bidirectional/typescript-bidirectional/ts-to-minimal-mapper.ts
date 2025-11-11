// Maps TypeScript AST to minimal typechecker expression format
import ts from 'typescript'

type Type =
  | { kind: 'number' }
  | { kind: 'string' }
  | { kind: 'function', from: Type, to: Type }

type Expr =
  | { kind: 'number', value: number }
  | { kind: 'string', value: string }
  | { kind: 'var', name: string }
  | { kind: 'lambda', param: string, body: Expr }
  | { kind: 'app', fn: Expr, arg: Expr }
  | { kind: 'let', name: string, value: Expr, body: Expr }

type Context = Map<string, Type>

// Synthesis: expr ⇒ type
function synth(ctx: Context, expr: Expr): Type {
  switch (expr.kind) {
    case 'number':
      return { kind: 'number' }

    case 'string':
      return { kind: 'string' }

    case 'var':
      const type = ctx.get(expr.name)
      if (!type) throw new Error(`Unbound variable: ${expr.name}`)
      return type

    case 'app':
      const fnType = synth(ctx, expr.fn)
      if (fnType.kind !== 'function') {
        throw new Error('Cannot apply non-function')
      }
      check(ctx, expr.arg, fnType.from)
      return fnType.to

    case 'lambda':
      throw new Error('Cannot synthesize type for lambda without annotation')

    case 'let':
      // Check if we have a pre-determined type for this binding (e.g., from function declaration)
      const expectedType = ctx.get(expr.name)
      let valueType: Type

      if (expectedType && expr.value.kind === 'lambda') {
        // Check lambda against expected type
        check(ctx, expr.value, expectedType)
        valueType = expectedType
      } else {
        valueType = synth(ctx, expr.value)
      }

      const newCtx = new Map(ctx)
      newCtx.set(expr.name, valueType)
      return synth(newCtx, expr.body)
  }
}

// Checking: expr ⇐ type
function check(ctx: Context, expr: Expr, expected: Type): void {
  switch (expr.kind) {
    case 'lambda':
      if (expected.kind !== 'function') {
        throw new Error('Lambda must have function type')
      }
      const newCtx = new Map(ctx)
      newCtx.set(expr.param, expected.from)
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
    return typesEqual(a.from, b.from) && typesEqual(a.to, b.to)
  }
  return true
}

function typeToString(type: Type): string {
  switch (type.kind) {
    case 'number': return 'number'
    case 'string': return 'string'
    case 'function': return `(${typeToString(type.from)} -> ${typeToString(type.to)})`
  }
}

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
        from: parseTypeAnnotation(paramType),
        to: parseTypeAnnotation(fnType.type)
      }
  }
  throw new Error(`Unsupported type annotation: ${ts.SyntaxKind[typeNode.kind]}`)
}

// Map TypeScript AST nodes to minimal Expr format
function mapTsNodeToExpr(node: ts.Node, ctx: Map<string, Type>): Expr {
  switch (node.kind) {
    case ts.SyntaxKind.NumericLiteral:
      const numLit = node as ts.NumericLiteral
      return { kind: 'number', value: parseFloat(numLit.text) }

    case ts.SyntaxKind.StringLiteral:
      const strLit = node as ts.StringLiteral
      return { kind: 'string', value: strLit.text }

    case ts.SyntaxKind.Identifier:
      const ident = node as ts.Identifier
      return { kind: 'var', name: ident.text }

    case ts.SyntaxKind.CallExpression:
      const call = node as ts.CallExpression
      return {
        kind: 'app',
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
        const returnStmt = func.body.statements.find(ts.isReturnStatement)
        if (!returnStmt || !returnStmt.expression) {
          throw new Error('Function must have return statement')
        }
        bodyExpr = mapTsNodeToExpr(returnStmt.expression, ctx)
      } else {
        bodyExpr = mapTsNodeToExpr(func.body, ctx)
      }

      return {
        kind: 'lambda',
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
          from: parseTypeAnnotation(param.type!),
          to: parseTypeAnnotation(init.type)
        }
        ctx.set(varName, funcType)
      }

      const valueExpr = mapTsNodeToExpr(decl.initializer!, ctx)

      // For variable statements, we need to wrap in a let expression
      // But we need the "body" - this will be handled at the program level
      return {
        kind: 'let',
        name: varName,
        value: valueExpr,
        body: { kind: 'var', name: varName } // placeholder
      }

    case ts.SyntaxKind.FunctionDeclaration:
      const funcDecl = node as ts.FunctionDeclaration
      const funcName = funcDecl.name!.text
      const funcParam = funcDecl.parameters[0]
      const funcParamName = (funcParam.name as ts.Identifier).text

      const returnStmt = (funcDecl.body as ts.Block).statements.find(ts.isReturnStatement)
      if (!returnStmt || !returnStmt.expression) {
        throw new Error('Function must have return statement')
      }

      const lambdaExpr: Expr = {
        kind: 'lambda',
        param: funcParamName,
        body: mapTsNodeToExpr(returnStmt.expression, ctx)
      }

      // Store function type in context
      const funcType: Type = {
        kind: 'function',
        from: parseTypeAnnotation(funcParam.type!),
        to: parseTypeAnnotation(funcDecl.type!)
      }
      ctx.set(funcName, funcType)

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
    // For non-let expressions, we can just continue to the next
    return buildProgramExpr(rest, ctx)
  }
}

// Main function to parse TypeScript code and typecheck with minimal typechecker
export function typecheckTypeScript(code: string): { expr: Expr, type: Type } {
  console.log(`\n=== TypeScript Code ===`)
  console.log(code)

  const sourceFile = ts.createSourceFile('test.ts', code, ts.ScriptTarget.Latest, true)
  const ctx = new Map<string, Type>()

  console.log(`\n=== Mapping to Minimal Expr ===`)
  const expr = buildProgramExpr(Array.from(sourceFile.statements), ctx)
  console.log(JSON.stringify(expr, null, 2))

  console.log(`\n=== Type Checking ===`)
  const type = synth(ctx, expr)
  console.log(`Type: ${typeToString(type)}`)

  return { expr, type }
}

// Demo
console.log('=== TypeScript to Minimal Typechecker Mapper ===\n')

// Example 1: Simple number
typecheckTypeScript('42')

// Example 2: Let binding
typecheckTypeScript(`
let x = 42
x
`.trim())

// Example 3: Function application
typecheckTypeScript(`
function id(x: number): number { return x }
id(5)
`.trim())

// Example 4: Multiple lets
typecheckTypeScript(`
let x = 10
let y = 20
y
`.trim())

// Example 5: Arrow function
typecheckTypeScript(`
const id = (x: number): number => x
id(42)
`.trim())

// Example 6: Nested function calls
typecheckTypeScript(`
function double(x: number): number { return x }
function add(x: number): number { return x }
add(double(5))
`.trim())
