import { typecheckTypeScript } from './ts-to-minimal-mapper'

console.log('=== Bidirectional Type Checker Demo ===\n')
console.log('A complex function body with multiple let bindings,')
console.log('function calls, and type annotations.\n')

typecheckTypeScript(`
// Helper functions (imagine they do the operations their names suggest)
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

// Main function: compute ((x * 2) + 10 - 3) then get string length
function compute(x: number): number {
  let doubled = multiply(x)(2)
  let withTen: number = add(doubled)(10)
  let subtracted = subtract(withTen)(3)
  let asString: string = toString(subtracted)
  return length(asString)
}

compute(5)
`)
