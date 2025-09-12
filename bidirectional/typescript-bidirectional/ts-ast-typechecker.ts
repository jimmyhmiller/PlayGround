// Minimal bidirectional type checker using TypeScript's AST
import ts from 'typescript'

type Type = 
  | { kind: 'number' }
  | { kind: 'string' }
  | { kind: 'boolean' }
  | { kind: 'function', from: Type, to: Type }
  | { kind: 'unit' }
  | { kind: 'unknown' }

type Context = Map<string, Type>

// Synthesis: node ⇒ type
function synth(ctx: Context, node: ts.Node): Type {
  switch (node.kind) {
    case ts.SyntaxKind.NumericLiteral:
      return { kind: 'number' }
    
    case ts.SyntaxKind.StringLiteral:
      return { kind: 'string' }
    
    case ts.SyntaxKind.TrueKeyword:
    case ts.SyntaxKind.FalseKeyword:
      return { kind: 'boolean' }
    
    case ts.SyntaxKind.Identifier:
      const name = (node as ts.Identifier).text
      return ctx.get(name)!
    
    case ts.SyntaxKind.CallExpression:
      const call = node as ts.CallExpression
      const fnType = synth(ctx, call.expression) as { kind: 'function', from: Type, to: Type }
      if (fnType.from.kind !== 'unit') {
        check(ctx, call.arguments[0], fnType.from)
      }
      return fnType.to
    
    case ts.SyntaxKind.FunctionDeclaration:
      const funcDecl = node as ts.FunctionDeclaration
      const param = funcDecl.parameters[0]
      
      const funcType: Type = {
        kind: 'function',
        from: param ? parseTypeAnnotation(param.type!) : { kind: 'unit' },
        to: parseTypeAnnotation(funcDecl.type!)
      }
      
      ctx.set(funcDecl.name!.text, funcType)
      
      const newCtx = new Map(ctx)
      if (param) {
        const paramName = (param.name as ts.Identifier).text
        newCtx.set(paramName, funcType.from)
      }
      
      const returnStmt = (funcDecl.body as ts.Block).statements.find(ts.isReturnStatement)!
      check(newCtx, returnStmt.expression!, funcType.to)
      
      return funcType
    
    case ts.SyntaxKind.VariableStatement:
      const varStmt = node as ts.VariableStatement
      const decl = varStmt.declarationList.declarations[0]
      const varName = (decl.name as ts.Identifier).text
      const valueType = synth(ctx, decl.initializer!)
      ctx.set(varName, valueType)
      return valueType
    
    case ts.SyntaxKind.ExpressionStatement:
      const exprStmt = node as ts.ExpressionStatement
      return synth(ctx, exprStmt.expression)
    
    case ts.SyntaxKind.FunctionExpression:
      const funcExpr = node as ts.FunctionExpression
      const exprParam = funcExpr.parameters[0]
      
      const exprFuncType: Type = {
        kind: 'function',
        from: parseTypeAnnotation(exprParam.type!),
        to: parseTypeAnnotation(funcExpr.type!)
      }
      
      const exprParamName = (exprParam.name as ts.Identifier).text
      const exprNewCtx = new Map(ctx)
      exprNewCtx.set(exprParamName, exprFuncType.from)
      
      const exprReturnStmt = (funcExpr.body as ts.Block).statements.find(ts.isReturnStatement)!
      check(exprNewCtx, exprReturnStmt.expression!, exprFuncType.to)
      
      return exprFuncType
  }
  
  return { kind: 'unknown' }
}

// Checking: node ⇐ type
function check(ctx: Context, node: ts.Node, expected: Type): void {
  const actual = synth(ctx, node)
  if (!typesEqual(actual, expected)) {
    throw new Error(`Type mismatch`)
  }
}

function parseTypeAnnotation(typeNode: ts.TypeNode): Type {
  switch (typeNode.kind) {
    case ts.SyntaxKind.NumberKeyword:
      return { kind: 'number' }
    case ts.SyntaxKind.StringKeyword:
      return { kind: 'string' }
    case ts.SyntaxKind.BooleanKeyword:
      return { kind: 'boolean' }
    case ts.SyntaxKind.FunctionType:
      const fnType = typeNode as ts.FunctionTypeNode
      const paramType = fnType.parameters[0].type!
      return {
        kind: 'function',
        from: parseTypeAnnotation(paramType),
        to: parseTypeAnnotation(fnType.type)
      }
  }
  return null as any
}

function typesEqual(a: Type, b: Type): boolean {
  if (a.kind !== b.kind) return false
  if (a.kind === 'function' && b.kind === 'function') {
    return typesEqual(a.from, b.from) && typesEqual(a.to, b.to)
  }
  return true
}


function checkProgram(code: string): void {
  console.log(`\nProgram: ${code}`)
  const sourceFile = ts.createSourceFile('test.ts', code, ts.ScriptTarget.Latest, true)
  const ctx = new Map<string, Type>()
  
  for (const stmt of sourceFile.statements) {
    const type = synth(ctx, stmt)
    console.log(`  ${stmt.getText(sourceFile)} ⇒ ${type.kind}`)
  }
}


// Demo
console.log('=== TypeScript AST Bidirectional Type Checker ===')

checkProgram('42')
checkProgram('"hello"')
checkProgram('true')
checkProgram('let x = 42')
checkProgram('let name = "hello"')
checkProgram('function id(x: number): number { return x }')

checkProgram(`
let x = 42
let y = "hello"
function id(x: number): number { return x }
id(x)
`)

checkProgram(`
function double(x: number): number { return x }
let num = 10
double(num)
`)

console.log('\n=== Most Complex Example ===')
checkProgram(`
function identity(x: number): number { return x }
function constant(x: number): number { return 42 }
function twice(x: number): number { return identity(identity(x)) }
function chain(x: number): number { return constant(twice(identity(x))) }
let a = 1
let b = 2
let c = 3
let step1 = identity(a)
let step2 = twice(b)  
let step3 = constant(c)
let step4 = chain(step1)
let final = identity(step4)
final
`)

console.log('\n=== Negative Tests ===')

// Type mismatch in function body
console.log('\nType mismatch in function body:')
try {
  checkProgram('function bad(x: number): string { return x }')
} catch (e) {
  console.log(`  ✓ Caught: ${e.message}`)
}

// Type mismatch in function call
console.log('\nType mismatch in function call:')
try {
  checkProgram(`
    function takesNumber(x: number): number { return x }
    takesNumber("hello")
  `)
} catch (e) {
  console.log(`  ✓ Caught: ${e.message}`)
}

// Calling non-function
console.log('\nCalling non-function:')
try {
  checkProgram(`
    let x = 42
    x(5)
  `)
} catch (e) {
  console.log(`  ✓ Caught: ${e.message}`)
}

// Undefined variable
console.log('\nUndefined variable:')
try {
  checkProgram('unknownVariable')
} catch (e) {
  console.log(`  ✓ Caught: ${e.message}`)
}

// Function parameter type mismatch
console.log('\nFunction parameter type mismatch:')
try {
  checkProgram(`
    function stringFunc(x: string): string { return x }
    let num = 42
    stringFunc(num)
  `)
} catch (e) {
  console.log(`  ✓ Caught: ${e.message}`)
}

// Zero parameter function test
console.log('\nZero parameter function:')
try {
  checkProgram(`
    function getValue(): number { return 42 }
    let result = getValue()
    result
  `)
} catch (e) {
  console.log(`  ✗ Failed: ${e.message}`)
}

// Wrong return type with variable
console.log('\nWrong return type with variable:')
try {
  checkProgram(`
    function shouldReturnString(x: number): string { 
      let y = x
      return y 
    }
  `)
} catch (e) {
  console.log(`  ✓ Caught: ${e.message}`)
}

// Complex type mismatch chain
console.log('\nComplex type mismatch chain:')
try {
  checkProgram(`
    function identity(x: number): number { return x }
    function badChain(x: string): number { return identity(x) }
    badChain("hello")
  `)
} catch (e) {
  console.log(`  ✓ Caught: ${e.message}`)
}