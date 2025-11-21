import * as ohm from 'ohm-js'
import { toAST } from 'ohm-js/extras'
import { synth, check } from './minimal-typechecker'
import type { Expr, Type, Context } from './minimal-typechecker'

const grammar = ohm.grammar(`
  MinimalLang {
    Program = Expr | Stmt

    Type = "(" Type ")" "=>" Type  -- fn
         | "number" | "string"

    Expr = "let" let_ident (":" Type)? "=" Expr ";" Expr  -- let
         | CallExpr

    CallExpr = CallExpr "(" Expr ")"  -- call
             | PrimaryExpr

    PrimaryExpr = number
                | stringLit
                | ident

    Block = "{" Stmt* "return" Expr ";" "}"

    Stmt = FnStmt
         | LetStmt
         | ExprStmt

    FnStmt = "function" let_ident "(" let_ident ":" Type ")" ":" Type Block

    LetStmt = "let" let_ident (":" Type)? "=" Expr ";"

    ExprStmt = CallExpr ";"

    number = digit+ ("." digit+)?
    stringLit = "\\"" (~"\\"" any)* "\\""
    ident = ~("function" | "let" | "return") letter (alnum | "_")*
    let_ident = letter (alnum | "_")*
  }
`)

const mapping = {
  Program: 0,
  Type(t) {
    const val = t.toAST(mapping);
    return typeof val === 'string' ? { kind: val } : val;
  },
  Stmt: 0,
  Expr_let: { type: undefined,  kind: 'let', name: 1, type: 3, value: 5},
  Type_fn: { type: undefined, kind: 'function', arg: 1, returnType: 4 },
  Expr_lambda: { type: undefined, kind: 'function', param: 1, body: 6 },
  CallExpr_call: { type: undefined, kind: 'call', fn: 0, arg: 2 },
  FnStmt: { kind: 'function', name: 1, param: 3, arg_type: 5, return_type: 8, body: 9 },
  LetStmt: { kind: 'let', name: 1, type: 3, value: 5 },
  ExprStmt: 0,
  number(_a, _b, _c) { return { kind: 'number', value: parseFloat(this.sourceString) } },
  stringLit(_q1, _chars, _q2) { return { kind: 'string', value: this.sourceString.slice(1, -1) } },
  ident(_a, _b) { return { kind: 'varLookup', name: this.sourceString } },
  Block: { type: undefined, kind: "block", statements: 1, return: 3},
}

export function parse(code: string): { expr: Expr, type: Type } {
  const match = grammar.match(code)
  if (match.failed()) throw new Error(match.message)
  const expr = toAST(match, mapping)
  return expr;
}

// Wrapper around synth that handles functions with type annotations
function synthWrapper(ctx: Context, expr: any): Type {
  // Special handling for functions with arg_type and return_type
  if (expr.kind === 'function' && expr.arg_type && expr.return_type) {
    // Build the function type signature
    const fnType: Type = {
      kind: 'function',
      arg: expr.arg_type,
      returnType: expr.return_type
    }

    // Use check instead of synth
    check(ctx, expr, fnType)
    return fnType
  }

  // For all other cases, use regular synth
  return synth(ctx, expr as Expr)
}

// Demo
import { inspect } from 'util'

console.log('=== Type Checker Compatibility Analysis ===\n')

const testCases = [
  // Basic primitives
  { name: '1. Number literal', code: '42' },
  { name: '2. String literal', code: '"hello"' },

  // Let expressions
  { name: '3. Let (no type)', code: 'let x = 10; x' },
  { name: '4. Let with type', code: 'let x: number = 42; x' },
  { name: '5. Nested lets', code: 'let x = 1; let y = 2; let z = 3; z' },
  { name: '6. Let with typed value', code: 'let x: number = 10; let y: string = "test"; y' },

  // Functions
  { name: '7. Simple function', code: 'function id(x: number): number { return x; }' },
  { name: '8. Function with block statements', code: 'function f(x: number): number { let y = 2; return x; }' },
  { name: '9. Function with multiple stmts', code: 'function g(x: number): number { let a = 1; let b = 2; let c = 3; return x; }' },

  // Function calls
  { name: '10. Simple call', code: 'f(5)', ctx: { f: { kind: 'function', arg: { kind: 'number' }, returnType: { kind: 'number' } } } },
  { name: '11. Nested calls', code: 'f(g(42))', ctx: {
    f: { kind: 'function', arg: { kind: 'number' }, returnType: { kind: 'number' } },
    g: { kind: 'function', arg: { kind: 'number' }, returnType: { kind: 'number' } }
  }},

  // Let with function calls
  { name: '12. Let with call', code: 'let x = f(5); x', ctx: {
    f: { kind: 'function', arg: { kind: 'number' }, returnType: { kind: 'number' } }
  }},

  // Higher-order functions
  { name: '13. Function returning function', code: 'function h(x: number): (number) => number { return f; }', ctx: {
    f: { kind: 'function', arg: { kind: 'number' }, returnType: { kind: 'number' } }
  }},

  // Complex let chains in functions
  { name: '14. Function with let chain', code: 'function chain(x: number): number { let y = x; let z = y; return z; }' },

  // String functions
  { name: '15. String identity', code: 'function strId(s: string): string { return s; }' },
  { name: '16. String with let', code: 'function strFn(s: string): string { let t = "hello"; return s; }' },

  // Mixed types in blocks
  { name: '17. Mixed types in block', code: 'function mixed(x: number): number { let s = "test"; let n = 42; return x; }' },

  // Function with call
  { name: '18. Function with call', code: 'function double(x: number): number { return x; }' },
]

for (const test of testCases) {
  console.log(`\n${'='.repeat(60)}`)
  console.log(`${test.name}: ${test.code}`)
  console.log('='.repeat(60))

  const parsed = parse(test.code) as any
  console.log('\nParsed AST:', inspect(parsed, { depth: null, colors: true }))

  // Build context from test definition
  const ctx = new Map()
  if (test.ctx) {
    for (const [name, type] of Object.entries(test.ctx)) {
      ctx.set(name, type as Type)
    }
  }

  try {
    const result = synthWrapper(ctx, parsed)
    console.log('\n✓ Type check SUCCEEDED')
    console.log('  Result type:', inspect(result, { depth: null, colors: true }))
  } catch (e) {
    console.log('\n✗ Type check FAILED')
    console.log('  Error:', (e as Error).message)
    console.log('  Stack:', (e as Error).stack)
  }
}

console.log('\n' + '='.repeat(60))
console.log('TEST SUMMARY')
console.log('='.repeat(60))
