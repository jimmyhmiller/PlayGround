import * as ohm from 'ohm-js'
import * as fs from 'fs'
import * as path from 'path'
import { fileURLToPath } from 'url'
import type { Expr, Type } from './minimal-typechecker'
import { synth, check } from './minimal-typechecker'

// Load the grammar
const __filename = fileURLToPath(import.meta.url)
const __dirname = path.dirname(__filename)
const grammarPath = path.join(__dirname, 'minimal-grammar.ohm')
const grammarSource = fs.readFileSync(grammarPath, 'utf-8')
const grammar = ohm.grammar(grammarSource)

// Semantic actions to convert parse tree to our AST
const semantics = grammar.createSemantics().addOperation('toAST', {
  Program(stmts, expr) {
    const statements = stmts.children.map(s => s.toAST())
    const returnExpr = expr.toAST()

    if (statements.length === 0) {
      return returnExpr
    }

    return {
      kind: 'block',
      statements,
      return: returnExpr
    }
  },

  LetStmt(_let, ident, typeAnnotation, _eq, expr, _semi) {
    const name = ident.sourceString
    const value = expr.toAST()
    const type = typeAnnotation.children.length > 0
      ? typeAnnotation.children[0].toAST()
      : undefined

    return {
      kind: 'let',
      name,
      value,
      type
    }
  },

  TypeAnnotation(_colon, type) {
    return type.toAST()
  },

  Type(type) {
    return type.toAST()
  },

  FunctionType(_lparen, argType, _rparen, _arrow, returnType) {
    return {
      kind: 'function',
      arg: argType.toAST(),
      returnType: returnType.toAST()
    }
  },

  PrimaryType(type) {
    const typeName = this.sourceString
    if (typeName === 'number') {
      return { kind: 'number' }
    } else if (typeName === 'string') {
      return { kind: 'string' }
    }
    throw new Error(`Unknown type: ${typeName}`)
  },

  LambdaExpr(_lparen, param, _colon, type, _rparen, _arrow, body) {
    return {
      kind: 'function',
      param: param.sourceString,
      body: body.toAST()
    }
  },

  CallExpr_call(fn, _lparen, arg, _rparen) {
    return {
      kind: 'call',
      fn: fn.toAST(),
      arg: arg.toAST()
    }
  },

  PrimaryExpr_paren(_lparen, expr, _rparen) {
    return expr.toAST()
  },

  PrimaryExpr(expr) {
    return expr.toAST()
  },

  CallExpr(expr) {
    return expr.toAST()
  },

  number(intPart, _dot, fracPart) {
    return {
      kind: 'number',
      value: parseFloat(this.sourceString)
    }
  },

  stringLit(_open, chars, _close) {
    return {
      kind: 'string',
      value: this.sourceString.slice(1, -1) // Remove quotes
    }
  },

  ident(_first, _rest) {
    return {
      kind: 'varLookup',
      name: this.sourceString
    }
  },

  _terminal() {
    return this.sourceString
  },

  _iter(...children) {
    return children.map(c => c.toAST())
  }
})

export function parseAndTypeCheck(source: string): { expr: Expr, type: Type } {
  console.log('=== Source Code ===')
  console.log(source)
  console.log()

  const match = grammar.match(source)

  if (match.failed()) {
    throw new Error(`Parse error: ${match.message}`)
  }

  const expr = semantics(match).toAST() as Expr

  console.log('=== Parsed AST ===')
  console.log(JSON.stringify(expr, null, 2))
  console.log()

  const ctx = new Map()
  const type = synth(ctx, expr)

  console.log('=== Type ===')
  console.log(type)
  console.log()

  return { expr, type }
}
