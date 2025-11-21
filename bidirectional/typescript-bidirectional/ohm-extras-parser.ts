import * as ohm from 'ohm-js'
import { toAST } from 'ohm-js/extras'
import { synth } from './minimal-typechecker'
import type { Expr, Type } from './minimal-typechecker'

const grammar = ohm.grammar(`
  MinimalLang {
    Program = Statement* Expr
    Statement = "let" ident (":" Type)? "=" Expr ";"
    Type = "(" Type ")" "=>" Type  -- fn
         | "number" | "string"
    Expr = "(" ident ":" Type ")" "=>" Expr  -- lambda
         | Expr "(" Expr ")"  -- call
         | "(" Expr ")"  -- paren
         | number | stringLit | ident
    number = digit+ ("." digit+)?
    stringLit = "\\"" (~"\\"" any)* "\\""
    ident = letter (alnum | "_")*
  }
`)

// Mapping to transform the auto-generated AST into our minimal AST format
const mapping = {
  Program(node) {
    const stmts = node[0] || []
    const ret = node[1]
    return stmts.length ? { kind: 'block', statements: stmts, return: ret } : ret
  },
  Statement(node) {
    return {
      kind: 'let',
      name: node[1],
      value: node[4],
      type: node[2] ? node[2][1] : undefined
    }
  },
  Type_fn(node) {
    return { kind: 'function', arg: node[1], returnType: node[4] }
  },
  Type(node) {
    const str = node._input || node
    return { kind: str === 'number' || str === 'string' ? str : node }
  },
  Expr_lambda(node) {
    return { kind: 'function', param: node[1], body: node[6] }
  },
  Expr_call(node) {
    return { kind: 'call', fn: node[0], arg: node[2] }
  },
  Expr_paren(node) {
    return node[1]
  },
  number(node) {
    return { kind: 'number', value: parseFloat(node._input) }
  },
  stringLit(node) {
    return { kind: 'string', value: node._input.slice(1, -1) }
  },
  ident(node) {
    return { kind: 'varLookup', name: node._input }
  }
}

export function parse(code: string): { expr: Expr, type: Type } {
  const match = grammar.match(code)
  if (match.failed()) throw new Error(match.message)

  // Use toAST with mapping
  const ast = toAST(match, mapping)

  console.log('Transformed AST:', JSON.stringify(ast, null, 2))

  const expr = ast as any as Expr
  const type = synth(new Map(), expr)
  return { expr, type }
}
