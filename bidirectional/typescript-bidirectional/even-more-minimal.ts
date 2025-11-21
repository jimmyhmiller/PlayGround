// Even more minimal: Type inference without AST construction
// Just walk the code and infer types directly

type Type = { kind: 'number' | 'string' }
type Env = Map<string, Type>

function infer(code: string, env: Env = new Map()): Type {
  code = code.trim()

  // Literals
  if (/^\d+$/.test(code)) return { kind: 'number' }
  if (/^".*"$/.test(code)) return { kind: 'string' }

  // Variables
  if (/^[a-z]\w*$/i.test(code)) {
    const t = env.get(code)
    if (!t) throw new Error(`Unbound: ${code}`)
    return t
  }

  // Let: let x: T = v; rest
  const m = code.match(/^let\s+(\w+)\s*(?::\s*(\w+))?\s*=\s*(.+?);\s*(.+)$/s)
  if (m) {
    const [, name, annot, val, rest] = m
    const t = infer(val, env)
    if (annot && t.kind !== annot) throw new Error(`Type mismatch`)
    return infer(rest, new Map(env).set(name, t))
  }

  throw new Error(`Parse error`)
}

// Demo
console.log('=== Even More Minimal (~30 lines total!) ===\n')
console.log(infer('42'))
console.log(infer('let x = 10; x'))
console.log(infer('let x: number = 42; x'))
console.log(infer('let x = 5; let y = 10; y'))
try { infer('let x: string = 42; x') } catch { console.log('Error caught ✓') }
console.log('\n✨ No AST, no parser library, just type inference!')
