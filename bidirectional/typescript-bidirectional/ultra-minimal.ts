// Ultra-minimal: Type checker as evaluator
// Uses regex + eval-like approach instead of full parsing

import type { Type, Context } from './minimal-typechecker'

// Parse a simple expression into AST with minimal code
function parseExpr(code: string, ctx: Context): { expr: any, type: Type } {
  code = code.trim()

  // Number
  if (/^\d+$/.test(code)) {
    return { expr: { kind: 'number', value: parseInt(code) }, type: { kind: 'number' } }
  }

  // String
  if (/^".*"$/.test(code)) {
    return { expr: { kind: 'string', value: code.slice(1, -1) }, type: { kind: 'string' } }
  }

  // Variable lookup
  if (/^[a-z]\w*$/i.test(code)) {
    const type = ctx.get(code)
    if (!type) throw new Error(`Unbound: ${code}`)
    return { expr: { kind: 'varLookup', name: code }, type }
  }

  // Let statements: "let x = 5; let y = 10; y"
  const letMatch = code.match(/^let\s+(\w+)\s*(?::\s*(\w+))?\s*=\s*(.+?);\s*(.+)$/s)
  if (letMatch) {
    const [, name, typeAnnot, value, rest] = letMatch
    const { expr: valExpr, type: valType } = parseExpr(value, ctx)

    // Check type annotation if present
    if (typeAnnot) {
      const expectedType = { kind: typeAnnot } as Type
      if (valType.kind !== typeAnnot) {
        throw new Error(`Type mismatch: expected ${typeAnnot}, got ${valType.kind}`)
      }
    }

    // Add to context and continue
    const newCtx = new Map(ctx)
    newCtx.set(name, valType)
    const { expr: restExpr, type: restType } = parseExpr(rest, newCtx)

    return {
      expr: { kind: 'block', statements: [{ kind: 'let', name, value: valExpr, type: typeAnnot ? { kind: typeAnnot } : undefined }], return: restExpr },
      type: restType
    }
  }

  throw new Error(`Cannot parse: ${code}`)
}

export function check(code: string): Type {
  const { type } = parseExpr(code, new Map())
  return type
}

// Demo
console.log('=== Ultra Minimal Parser (~50 lines) ===\n')
console.log('1.', check('42'))
console.log('2.', check('let x = 10; x'))
console.log('3.', check('let x: number = 42; x'))
console.log('4.', check('let x = 5; let y = 10; y'))

try {
  check('let x: string = 42; x')
} catch (e) {
  console.log('5. Type error caught! ✓')
}

console.log('\n✨ Using regex + recursion instead of full parser!')
