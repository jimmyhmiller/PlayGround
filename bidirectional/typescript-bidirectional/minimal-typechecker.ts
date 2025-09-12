// Ultra-minimal bidirectional type checker
// Demonstrates core concepts with zero indirection

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
      const valueType = synth(ctx, expr.value)
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

// Examples that work:
const examples = {
  // Numbers and strings
  num: { kind: 'number', value: 42 } as Expr,
  str: { kind: 'string', value: 'hello' } as Expr,
  
  // Identity function with number
  identity: {
    kind: 'lambda',
    param: 'x',
    body: { kind: 'var', name: 'x' }
  } as Expr,
  
  // Function application
  app: {
    kind: 'app',
    fn: { kind: 'lambda', param: 'x', body: { kind: 'var', name: 'x' } },
    arg: { kind: 'number', value: 5 }
  } as Expr,
  
  // Let binding
  letExpr: {
    kind: 'let',
    name: 'x',
    value: { kind: 'number', value: 10 },
    body: { kind: 'var', name: 'x' }
  } as Expr
}

// Demo
console.log('=== Minimal Bidirectional Type Checker ===\n')

// Synthesize types
console.log('Synthesis (⇒):')
console.log(`42 ⇒ ${typeToString(synth(new Map(), examples.num))}`)
console.log(`"hello" ⇒ ${typeToString(synth(new Map(), examples.str))}`)
console.log(`let x = 10 in x ⇒ ${typeToString(synth(new Map(), examples.letExpr))}\n`)

// Check against types
console.log('Checking (⇐):')
const numType = { kind: 'number' } as Type
const fnType = { kind: 'function', from: numType, to: numType } as Type

try {
  check(new Map(), examples.identity, fnType)
  console.log(`λx.x ⇐ (number -> number) ✓`)
} catch (e) {
  console.log(`λx.x ⇐ (number -> number) ✗`)
}

try {
  check(new Map(), examples.app, numType)
  console.log(`(λx.x) 5 ⇐ number ✓`)
} catch (e) {
  console.log(`(λx.x) 5 ⇐ number ✗`)
}

// Error example
console.log('\nError handling:')
try {
  const badApp = {
    kind: 'app',
    fn: { kind: 'number', value: 42 },
    arg: { kind: 'string', value: 'hello' }
  } as Expr
  synth(new Map(), badApp)
} catch (e) {
  console.log(`42("hello") → Error: ${(e as Error).message}`)
}