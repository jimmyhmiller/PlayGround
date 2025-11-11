// Extended minimal bidirectional type checker
// Supports multiple parameters, if expressions, binary operations, void type

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

// Synthesis: expr ⇒ type
function synth(ctx: Context, expr: Expr): Type {
  switch (expr.kind) {
    case 'number':
      return { kind: 'number' }

    case 'string':
      return { kind: 'string' }

    case 'boolean':
      return { kind: 'boolean' }

    case 'var':
      const type = ctx.get(expr.name)
      if (!type) throw new Error(`Unbound variable: ${expr.name}`)
      return type

    case 'app':
      const fnType = synth(ctx, expr.fn)
      if (fnType.kind !== 'function') {
        throw new Error('Cannot apply non-function')
      }

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

          // Support number + number or string + string
          if (leftType.kind === 'number' && rightType.kind === 'number') {
            return { kind: 'number' }
          }
          if (leftType.kind === 'string' && rightType.kind === 'string') {
            return { kind: 'string' }
          }
          // Support string + number or number + string (coercion)
          if ((leftType.kind === 'string' && rightType.kind === 'number') ||
              (leftType.kind === 'number' && rightType.kind === 'string')) {
            return { kind: 'string' }
          }

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
      if (expr.exprs.length === 0) {
        return { kind: 'void' }
      }

      // Check all expressions except the last
      for (let i = 0; i < expr.exprs.length - 1; i++) {
        synth(ctx, expr.exprs[i])
      }

      // Return type of the last expression
      return synth(ctx, expr.exprs[expr.exprs.length - 1])
  }
}

// Checking: expr ⇐ type
function check(ctx: Context, expr: Expr, expected: Type): void {
  switch (expr.kind) {
    case 'lambda':
      if (expected.kind !== 'function') {
        throw new Error('Lambda must have function type')
      }

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

// Examples that work:
const examples = {
  // Binary operations
  add: {
    kind: 'binop',
    op: '+' as const,
    left: { kind: 'number', value: 1 } as Expr,
    right: { kind: 'number', value: 2 } as Expr
  } as Expr,

  // String concatenation
  concat: {
    kind: 'binop',
    op: '+' as const,
    left: { kind: 'string', value: 'hello' } as Expr,
    right: { kind: 'string', value: 'world' } as Expr
  } as Expr,

  // If expression
  ifExpr: {
    kind: 'if',
    condition: {
      kind: 'binop',
      op: '>' as const,
      left: { kind: 'number', value: 5 } as Expr,
      right: { kind: 'number', value: 3 } as Expr
    } as Expr,
    then: { kind: 'number', value: 10 } as Expr,
    else: { kind: 'number', value: 20 } as Expr
  } as Expr,

  // Multiple parameters
  multiParam: {
    kind: 'lambda',
    params: ['x', 'y'],
    body: {
      kind: 'binop',
      op: '+' as const,
      left: { kind: 'var', name: 'x' } as Expr,
      right: { kind: 'var', name: 'y' } as Expr
    } as Expr
  } as Expr,

  // Sequence
  sequence: {
    kind: 'sequence',
    exprs: [
      { kind: 'number', value: 1 } as Expr,
      { kind: 'number', value: 2 } as Expr,
      { kind: 'number', value: 3 } as Expr
    ]
  } as Expr
}

// Demo
console.log('=== Extended Minimal Bidirectional Type Checker ===\n')

// Synthesize types
console.log('Synthesis (⇒):')
console.log(`1 + 2 ⇒ ${typeToString(synth(new Map(), examples.add))}`)
console.log(`"hello" + "world" ⇒ ${typeToString(synth(new Map(), examples.concat))}`)
console.log(`if (5 > 3) 10 else 20 ⇒ ${typeToString(synth(new Map(), examples.ifExpr))}`)
console.log(`sequence [1, 2, 3] ⇒ ${typeToString(synth(new Map(), examples.sequence))}\n`)

// Check against types
console.log('Checking (⇐):')
const numType = { kind: 'number' } as Type
const multiParamFnType = {
  kind: 'function',
  params: [numType, numType],
  to: numType
} as Type

try {
  check(new Map(), examples.multiParam, multiParamFnType)
  console.log(`λ(x, y). x + y ⇐ (number, number) -> number ✓`)
} catch (e) {
  console.log(`λ(x, y). x + y ⇐ (number, number) -> number ✗: ${(e as Error).message}`)
}

// Complex example with let bindings
console.log('\nComplex example:')
const complexExpr: Expr = {
  kind: 'let',
  name: 'add',
  value: {
    kind: 'lambda',
    params: ['x', 'y'],
    body: {
      kind: 'binop',
      op: '+',
      left: { kind: 'var', name: 'x' },
      right: { kind: 'var', name: 'y' }
    }
  },
  body: {
    kind: 'app',
    fn: { kind: 'var', name: 'add' },
    args: [
      { kind: 'number', value: 10 },
      { kind: 'number', value: 20 }
    ]
  }
}

const ctx = new Map<string, Type>()
ctx.set('add', multiParamFnType)
try {
  const result = synth(ctx, complexExpr)
  console.log(`let add = λ(x,y).x+y in add(10, 20) ⇒ ${typeToString(result)} ✓`)
} catch (e) {
  console.log(`Error: ${(e as Error).message}`)
}

// Void type example
console.log('\nVoid type:')
const voidFnType: Type = {
  kind: 'function',
  params: [{ kind: 'string' }],
  to: { kind: 'void' }
}

const logExpr: Expr = {
  kind: 'lambda',
  params: ['msg'],
  body: {
    kind: 'sequence',
    exprs: []  // Empty sequence = void
  }
}

const ctx2 = new Map<string, Type>()
ctx2.set('log', voidFnType)
try {
  check(ctx2, logExpr, voidFnType)
  console.log(`λ(msg). void ⇐ (string) -> void ✓`)
} catch (e) {
  console.log(`Error: ${(e as Error).message}`)
}
