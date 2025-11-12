// Pure data transformation: TypeScript AST -> Minimal2 Expr format
// No typechecking happens here - just AST transformation
import ts from 'typescript'
import { Type, Expr, Context, synth, typeToString } from './minimal2-typechecker.js'

// Helper type to track type annotations we extract from the AST
type TypeAnnotations = Map<string, Type>

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

// Pure transformation: TypeScript AST node -> Expr
// The annotations map is only READ to attach type info, never modified
function mapTsNodeToExpr(node: ts.Node, annotations: TypeAnnotations): Expr {
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
        fn: mapTsNodeToExpr(call.expression, annotations),
        args: call.arguments.map(arg => mapTsNodeToExpr(arg, annotations))
      }

    case ts.SyntaxKind.BinaryExpression:
      const binExpr = node as ts.BinaryExpression
      return {
        kind: 'binop',
        op: mapBinaryOp(binExpr.operatorToken.kind),
        left: mapTsNodeToExpr(binExpr.left, annotations),
        right: mapTsNodeToExpr(binExpr.right, annotations)
      }

    case ts.SyntaxKind.IfStatement:
      const ifStmt = node as ts.IfStatement
      return {
        kind: 'if',
        condition: mapTsNodeToExpr(ifStmt.expression, annotations),
        then: mapTsNodeToExpr(ifStmt.thenStatement, annotations),
        else: ifStmt.elseStatement
          ? mapTsNodeToExpr(ifStmt.elseStatement, annotations)
          : { kind: 'sequence', exprs: [] }
      }

    case ts.SyntaxKind.Block:
      const block = node as ts.Block
      return mapBlockToExpr(block, annotations)

    case ts.SyntaxKind.ReturnStatement:
      const returnStmt = node as ts.ReturnStatement
      return returnStmt.expression
        ? mapTsNodeToExpr(returnStmt.expression, annotations)
        : { kind: 'sequence', exprs: [] }

    case ts.SyntaxKind.ArrowFunction:
    case ts.SyntaxKind.FunctionExpression:
      const func = node as ts.ArrowFunction | ts.FunctionExpression
      const params = func.parameters.map(p => (p.name as ts.Identifier).text)
      let bodyExpr: Expr
      if (ts.isBlock(func.body)) {
        bodyExpr = mapBlockToExpr(func.body, annotations)
      } else {
        bodyExpr = mapTsNodeToExpr(func.body, annotations)
      }
      return { kind: 'lambda', params, body: bodyExpr }

    case ts.SyntaxKind.VariableStatement:
      const varStmt = node as ts.VariableStatement
      const decl = varStmt.declarationList.declarations[0]
      const varName = (decl.name as ts.Identifier).text
      const valueExpr = mapTsNodeToExpr(decl.initializer!, annotations)
      return { kind: 'let', name: varName, value: valueExpr, body: { kind: 'var', name: varName } }

    case ts.SyntaxKind.FunctionDeclaration:
      const funcDecl = node as ts.FunctionDeclaration
      const funcName = funcDecl.name!.text
      const funcParams = funcDecl.parameters.map(p => (p.name as ts.Identifier).text)
      const lambdaExpr: Expr = {
        kind: 'lambda',
        params: funcParams,
        body: mapBlockToExpr(funcDecl.body!, annotations)
      }
      return { kind: 'let', name: funcName, value: lambdaExpr, body: { kind: 'var', name: funcName } }

    case ts.SyntaxKind.ExpressionStatement:
      return mapTsNodeToExpr((node as ts.ExpressionStatement).expression, annotations)
  }

  throw new Error(`Unsupported node kind: ${ts.SyntaxKind[node.kind]}`)
}

// Map a block statement to an expression
function mapBlockToExpr(block: ts.Block, annotations: TypeAnnotations): Expr {
  const statements = Array.from(block.statements)
  if (statements.length === 0) return { kind: 'sequence', exprs: [] }
  return mapStatementsToExpr(statements, annotations)
}

// Transform a sequence of statements into nested let expressions
function mapStatementsToExpr(statements: ts.Statement[], annotations: TypeAnnotations): Expr {
  if (statements.length === 0) return { kind: 'sequence', exprs: [] }

  const [first, ...rest] = statements

  // Handle variable declarations
  if (ts.isVariableStatement(first)) {
    const decl = first.declarationList.declarations[0]
    const varName = (decl.name as ts.Identifier).text
    const valueExpr = mapTsNodeToExpr(decl.initializer!, annotations)
    const bodyExpr = rest.length > 0
      ? mapStatementsToExpr(rest, annotations)
      : { kind: 'sequence', exprs: [] }
    return { kind: 'let', name: varName, value: valueExpr, body: bodyExpr }
  }

  // Handle return statements
  if (ts.isReturnStatement(first)) {
    return first.expression
      ? mapTsNodeToExpr(first.expression, annotations)
      : { kind: 'sequence', exprs: [] }
  }

  // Handle if statements with early returns
  if (ts.isIfStatement(first)) {
    const ifExpr = mapTsNodeToExpr(first, annotations)
    if (rest.length > 0) {
      const hasReturnInThen = hasReturnStatement(first.thenStatement)
      const hasReturnInElse = first.elseStatement ? hasReturnStatement(first.elseStatement) : false

      if (hasReturnInThen && hasReturnInElse) {
        return ifExpr
      } else if (!hasReturnInThen && !hasReturnInElse) {
        const restExpr = mapStatementsToExpr(rest, annotations)
        return { kind: 'sequence', exprs: [ifExpr, restExpr] }
      } else {
        const restExpr = mapStatementsToExpr(rest, annotations)
        if (hasReturnInThen && !hasReturnInElse) {
          return {
            kind: 'if',
            condition: (ifExpr as any).condition,
            then: (ifExpr as any).then,
            else: (ifExpr as any).else.kind === 'sequence' && (ifExpr as any).else.exprs.length === 0
              ? restExpr
              : { kind: 'sequence', exprs: [(ifExpr as any).else, restExpr] }
          }
        } else {
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

  // Handle other statements
  const firstExpr = mapTsNodeToExpr(first, annotations)
  if (rest.length === 0) return firstExpr
  const restExpr = mapStatementsToExpr(rest, annotations)
  if (restExpr.kind === 'sequence' && restExpr.exprs.length === 0) return firstExpr
  return { kind: 'sequence', exprs: [firstExpr, restExpr] }
}

// Helper to check if a statement contains a return
function hasReturnStatement(stmt: ts.Statement): boolean {
  if (ts.isReturnStatement(stmt)) return true
  if (ts.isBlock(stmt)) return stmt.statements.some(s => hasReturnStatement(s))
  return false
}

// Extract type annotations from the TypeScript AST
// This is a separate pass that collects all the type information
function extractTypeAnnotations(sourceFile: ts.SourceFile): TypeAnnotations {
  const annotations = new Map<string, Type>()

  function visit(node: ts.Node) {
    // Function declarations
    if (ts.isFunctionDeclaration(node) && node.name) {
      const funcName = node.name.text
      const funcType: Type = {
        kind: 'function',
        params: node.parameters.map(p => parseTypeAnnotation(p.type)),
        to: parseTypeAnnotation(node.type)
      }
      annotations.set(funcName, funcType)
    }

    // Variable declarations with arrow/function expressions
    if (ts.isVariableStatement(node)) {
      const decl = node.declarationList.declarations[0]
      const varName = (decl.name as ts.Identifier).text
      const init = decl.initializer
      if (init && (ts.isArrowFunction(init) || ts.isFunctionExpression(init))) {
        if (init.type || init.parameters.some(p => p.type)) {
          const funcType: Type = {
            kind: 'function',
            params: init.parameters.map(p => parseTypeAnnotation(p.type)),
            to: parseTypeAnnotation(init.type)
          }
          annotations.set(varName, funcType)
        }
      }
    }

    ts.forEachChild(node, visit)
  }

  ts.forEachChild(sourceFile, visit)
  return annotations
}

// Convert a sequence of statements into nested let expressions
function buildProgramExpr(statements: ts.Statement[], annotations: TypeAnnotations): Expr {
  if (statements.length === 0) throw new Error('Empty program')
  if (statements.length === 1) return mapTsNodeToExpr(statements[0], annotations)

  const [first, ...rest] = statements
  const firstExpr = mapTsNodeToExpr(first, annotations)

  if (firstExpr.kind === 'let') {
    return { kind: 'let', name: firstExpr.name, value: firstExpr.value, body: buildProgramExpr(rest, annotations) }
  } else {
    return buildProgramExpr(rest, annotations)
  }
}

// Main function: Pure transformation then typecheck
export function typecheckTypeScript(code: string): { expr: Expr, type: Type, annotations: TypeAnnotations } {
  console.log(`\n=== TypeScript Code ===`)
  console.log(code)

  // Parse TypeScript
  const sourceFile = ts.createSourceFile('test.ts', code, ts.ScriptTarget.Latest, true)

  // Step 1: Extract type annotations (pure data extraction)
  console.log(`\n=== Extracting Type Annotations ===`)
  const annotations = extractTypeAnnotations(sourceFile)
  console.log(JSON.stringify(Array.from(annotations.entries()).map(([name, type]) => [name, typeToString(type)]), null, 2))

  // Step 2: Transform AST to Expr (pure transformation)
  console.log(`\n=== Mapping to Minimal2 Expr ===`)
  const expr = buildProgramExpr(Array.from(sourceFile.statements), annotations)
  console.log(JSON.stringify(expr, null, 2))

  // Step 3: Typecheck the Expr (separate step)
  console.log(`\n=== Type Checking ===`)
  const ctx = new Map(annotations)
  const type = synth(ctx, expr)
  console.log(`Type: ${typeToString(type)}`)

  return { expr, type, annotations }
}

// Demo
console.log('=== TypeScript to Minimal2 Typechecker Mapper (Pure) ===\n')

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
