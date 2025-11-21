import * as ohm from 'ohm-js'
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

const toAST = grammar.createSemantics().addOperation('ast', {
  Program(stmts, expr) {
    const statements = stmts.children.map(c => c.ast())
    const ret = expr.ast()
    return statements.length ? { kind: 'block', statements, return: ret } : ret
  },
  Statement(_let, name, _colon, type, _eq, val, _semi) {
    return { kind: 'let', name: name.sourceString, value: val.ast(),
             type: type.children[0]?.ast() }
  },
  Type_fn(_lp, arg, _rp, _arrow, ret) {
    return { kind: 'function', arg: arg.ast(), returnType: ret.ast() }
  },
  Type(t) {
    const s = this.sourceString
    return s === 'number' || s === 'string' ? { kind: s } : t.ast()
  },
  Expr_lambda(_lp, param, _colon, type, _rp, _arrow, body) {
    return { kind: 'function', param: param.sourceString, body: body.ast() }
  },
  Expr_call(fn, _lp, arg, _rp) {
    return { kind: 'call', fn: fn.ast(), arg: arg.ast() }
  },
  Expr_paren(_lp, e, _rp) { return e.ast() },
  Expr(e) { return e.ast() },
  number(_int, _dot, _frac) { return { kind: 'number', value: parseFloat(this.sourceString) } },
  stringLit(_q1, chars, _q2) { return { kind: 'string', value: this.sourceString.slice(1, -1) } },
  ident(_a, _b) { return { kind: 'varLookup', name: this.sourceString } },
})

export function parse(code: string): { expr: Expr, type: Type } {
  const match = grammar.match(code)
  if (match.failed()) throw new Error(match.message)
  const expr = toAST(match).ast()
  const type = synth(new Map(), expr)
  return { expr, type }
}
