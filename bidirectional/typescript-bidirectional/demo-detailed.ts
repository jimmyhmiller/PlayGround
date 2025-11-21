import { typecheckTypeScript } from './ts-to-minimal-mapper'

console.log('=== Bidirectional Type Checker Demo ===\n')
console.log('Showcasing a complex function body with:')
console.log('  • Multiple let bindings')
console.log('  • Higher-order function calls (curried functions)')
console.log('  • Mixed type annotations (some explicit, some inferred)')
console.log('  • Type flow from number → string → number\n')

typecheckTypeScript(`
// Helper functions (curried for composition)
function add(x: number): (y: number) => number {
  return (y: number): number => y
}

function multiply(x: number): (y: number) => number {
  return (y: number): number => y
}

function subtract(x: number): (y: number) => number {
  return (y: number): number => y
}

function toString(x: number): string {
  return "result"
}

function length(s: string): number {
  return 42
}

// Main computation: ((x * 2) + 10 - 3) as string length
function compute(x: number): number {
  let doubled = multiply(x)(2)
  let withTen: number = add(doubled)(10)
  let subtracted = subtract(withTen)(3)
  let asString: string = toString(subtracted)
  return length(asString)
}

compute(5)
`)

console.log('\n' + '='.repeat(60))
console.log('Key Features Demonstrated:')
console.log('='.repeat(60))
console.log('✓ Bidirectional type checking (⇒ synthesis / ⇐ checking)')
console.log('✓ Function type synthesis and checking')
console.log('✓ Higher-order functions (functions returning functions)')
console.log('✓ Let bindings with context mutation')
console.log('✓ Optional type annotations on let bindings')
console.log('✓ Type error detection on mismatches')
