// Ultra-minimal bidirectional type checker
// Demonstrates core concepts with zero indirection

export type Type =
  | { kind: 'number' }
  | { kind: 'string' }
  | { kind: 'function', arg: Type, returnType: Type }

export type Expr =
  | { kind: 'number', value: number }
  | { kind: 'string', value: string }
  | { kind: 'varLookup', name: string }
  | { kind: 'function', param: string, body: Expr }
  | { kind: 'call', fn: Expr, arg: Expr }
  | { kind: 'let', name: string, value: Expr, type?: Type }
  | { kind: 'block', statements: Expr[], return: Expr }

export type Context = Map<string, Type>

// Synthesis: expr ⇒ type
export function synth(ctx: Context, expr: Expr): Type {
  switch (expr.kind) {
    case 'number':
      return { kind: 'number' }
    
    case 'string':
      return { kind: 'string' }

    case 'varLookup':
      const type = ctx.get(expr.name)
      if (!type) throw new Error(`Unbound variable: ${expr.name}`)
      return type
    
    case 'call':
      const fnType = synth(ctx, expr.fn)
      if (fnType.kind !== 'function') {
        throw new Error('Cannot call non-function')
      }
      check(ctx, expr.arg, fnType.arg)
      return fnType.returnType
    
    case 'function':
      throw new Error('Cannot synthesize type for function without annotation')

    case 'let':
      // Synthesize type of the value
      const valueType = synth(ctx, expr.value)
      // If an explicit type is provided, check the value against it
      if (expr.type) {
        if (!typesEqual(valueType, expr.type)) {
          throw new Error(`Type mismatch in let binding: expected ${JSON.stringify(expr.type)}, got ${JSON.stringify(valueType)}`)
        }
      }
      // Add binding to context (side effect!)
      ctx.set(expr.name, valueType)
      // Return the type of the value
      return valueType

    case 'block':
      let blockCtx = new Map(ctx)
      for (const stmt of expr.statements) {
        synth(blockCtx, stmt)
      }
      return synth(blockCtx, expr.return)
  }
}

// Checking: expr ⇐ type
export function check(ctx: Context, expr: Expr, expected: Type): void {
  switch (expr.kind) {
    case 'function':
      if (expected.kind !== 'function') {
        throw new Error('Function must have function type')
      }
      const newCtx = new Map(ctx)
      newCtx.set(expr.param, expected.arg)
      check(newCtx, expr.body, expected.returnType)
      break

    case 'block':
      let blockCtx = new Map(ctx)
      for (const stmt of expr.statements) {
        synth(blockCtx, stmt)
      }
      check(blockCtx, expr.return, expected)
      break

    default:
      const actual = synth(ctx, expr)
      if (!typesEqual(actual, expected)) {
        throw new Error(`Type mismatch: expected ${JSON.stringify(expected)}, got ${JSON.stringify(actual)}`)
      }
  }
}

export function typesEqual(a: Type, b: Type): boolean {
  if (a.kind !== b.kind) return false
  if (a.kind === 'function' && b.kind === 'function') {
    return typesEqual(a.arg, b.arg) && typesEqual(a.returnType, b.returnType)
  }
  return true
}


function runDemo() {
  // Examples that work:
  const examples = {
    // Numbers and strings
    num: { kind: 'number', value: 42 } as Expr,
    str: { kind: 'string', value: 'hello' } as Expr,
    
    // Identity function with number
    identity: {
      kind: 'function',
      param: 'x',
      body: { kind: 'varLookup', name: 'x' }
    } as Expr,
    
    // Function application
    applyExpr: {
      kind: 'call',
      fn: { kind: 'varLookup', name: 'id' },
      arg: { kind: 'number', value: 5 }
    } as Expr,
    
    // Let binding in a block
    letExpr: {
      kind: 'block',
      statements: [
        { kind: 'let', name: 'x', value: { kind: 'number', value: 10 } }
      ],
      return: { kind: 'varLookup', name: 'x' }
    } as Expr,

    // Let binding with explicit type annotation
    letWithType: {
      kind: 'block',
      statements: [
        { kind: 'let', name: 'y', value: { kind: 'number', value: 42 }, type: { kind: 'number' } }
      ],
      return: { kind: 'varLookup', name: 'y' }
    } as Expr
  }

  const numType: Type = { kind: 'number' }
  const stringType: Type = { kind: 'string' }
  const fnType: Type = { kind: 'function', arg: numType, returnType: numType }

  const ctxWithId: Context = new Map()
  ctxWithId.set('id', fnType)

  const extraExprs = {
    wrongArgApply: {
      kind: 'call',
      fn: { kind: 'varLookup', name: 'id' },
      arg: { kind: 'string', value: 'oops' }
    } as Expr,
    outerLetNumber: {
      kind: 'block',
      statements: [
        { kind: 'let', name: 'x', value: { kind: 'number', value: 21 } },
        { kind: 'let', name: 'y', value: { kind: 'string', value: 'buffer' } }
      ],
      return: { kind: 'varLookup', name: 'x' }
    } as Expr,
    shadowedLetString: {
      kind: 'block',
      statements: [
        { kind: 'let', name: 'x', value: { kind: 'number', value: 11 } },
        { kind: 'let', name: 'x', value: { kind: 'string', value: 'inner' } }
      ],
      return: { kind: 'varLookup', name: 'x' }
    } as Expr,
    unboundVar: { kind: 'varLookup', name: 'ghost' } as Expr,
    wrongTypeAnnotation: {
      kind: 'block',
      statements: [
        { kind: 'let', name: 'z', value: { kind: 'number', value: 100 }, type: { kind: 'string' } }
      ],
      return: { kind: 'varLookup', name: 'z' }
    } as Expr
  }

  // Demo
  console.log('=== Minimal Bidirectional Type Checker ===\n')

  // Synthesize types
  console.log('Synthesis (⇒):')
  console.log(`42 ⇒`, synth(new Map(), examples.num))
  console.log(`"hello" ⇒`, synth(new Map(), examples.str))
  console.log(`let x = 10 in x ⇒`, synth(new Map(), examples.letExpr))
  console.log(`let y: number = 42 in y ⇒`, synth(new Map(), examples.letWithType))
  console.log(`id 5 ⇒`, synth(ctxWithId, examples.applyExpr))
  console.log(`let x = 21; let y = "buffer"; x ⇒`, synth(new Map(), extraExprs.outerLetNumber))
  console.log(`let x = 11; let x = "inner"; x ⇒`, synth(new Map(), extraExprs.shadowedLetString))
  console.log()

  // Check against types
  console.log('Checking (⇐):')
  try {
    check(new Map(), examples.identity, fnType)
    console.log(`function(x) ⇐ (number -> number) ✓`)
  } catch (e) {
    console.log(`function(x) ⇐ (number -> number) ✗ → ${(e as Error).message}`)
  }

  try {
    check(ctxWithId, examples.applyExpr, numType)
    console.log(`id 5 ⇐ number ✓`)
  } catch (e) {
    console.log(`id 5 ⇐ number ✗ → ${(e as Error).message}`)
  }

  try {
    check(new Map(), extraExprs.outerLetNumber, numType)
    console.log(`let x=21; let y="buffer"; x ⇐ number ✓`)
  } catch (e) {
    console.log(`let x=21; let y="buffer"; x ⇐ number ✗ → ${(e as Error).message}`)
  }

  try {
    check(new Map(), extraExprs.shadowedLetString, stringType)
    console.log(`shadowed let returning string ⇐ string ✓`)
  } catch (e) {
    console.log(`shadowed let returning string ⇐ string ✗ → ${(e as Error).message}`)
  }

  try {
    check(ctxWithId, extraExprs.wrongArgApply, numType)
    console.log(`id "oops" ⇐ number ✓ (unexpected)`)
  } catch (e) {
    console.log(`id "oops" ⇐ number ✗ → ${(e as Error).message}`)
  }

  try {
    check(new Map(), extraExprs.shadowedLetString, numType)
    console.log(`shadowed let returning string ⇐ number ✓ (unexpected)`)
  } catch (e) {
    console.log(`shadowed let returning string ⇐ number ✗ → ${(e as Error).message}`)
  }

  // Error example
  console.log('\nError handling:')
  try {
    const badApp = {
      kind: 'call',
      fn: { kind: 'number', value: 42 },
      arg: { kind: 'string', value: 'hello' }
    } as Expr
    synth(new Map(), badApp)
  } catch (e) {
    console.log(`42("hello") → Error: ${(e as Error).message}`)
  }

  try {
    synth(new Map(), extraExprs.unboundVar)
    console.log(`ghost ⇒ (unexpected success)`)
  } catch (e) {
    console.log(`ghost ⇒ Error: ${(e as Error).message}`)
  }

  try {
    synth(new Map(), extraExprs.wrongTypeAnnotation)
    console.log(`let z: string = 100 in z ⇒ (unexpected success)`)
  } catch (e) {
    console.log(`let z: string = 100 in z ⇒ Error: ${(e as Error).message}`)
  }
}

const maybeProcess = (globalThis as any)?.process
const argvList: unknown = maybeProcess?.argv
const isDirectExecution =
  Array.isArray(argvList) &&
  argvList.some(arg =>
    typeof arg === 'string' && /minimal-typechecker(\.ts|\.js)?$/.test(arg)
  )

if (isDirectExecution) {
  runDemo()
}
