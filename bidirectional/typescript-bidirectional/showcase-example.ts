import { typecheckTypeScript } from './ts-to-minimal-mapper'

console.log('=== SHOWCASE: Complex TypeScript Type Checking ===\n')

// Example 1: Function composition with let bindings and type annotations
console.log('Example 1: Function Pipeline with Type Annotations')
typecheckTypeScript(`
function double(x: number): number {
  return x
}

function addTen(x: number): number {
  return x
}

let x: number = 5
let doubled = double(x)
let result: number = addTen(doubled)
result
`)

// Example 2: Higher-order functions
console.log('\nExample 2: Higher-Order Function')
typecheckTypeScript(`
function makeAdder(x: number): (y: number) => number {
  return (y: number): number => y
}

const add5 = makeAdder(5)
add5(10)
`)

// Example 3: Multiple function applications with variable shadowing
console.log('\nExample 3: Variable Shadowing')
typecheckTypeScript(`
function transform(x: number): number { return x }

let x = 10
let y = transform(x)
let x = 20
transform(x)
`)

// Example 4: Type checking catches errors
console.log('\nExample 4: Type Error Detection')
try {
  typecheckTypeScript(`
function expectsNumber(x: number): number { return x }

let value: number = 42
let wrongValue: string = "hello"
expectsNumber(wrongValue)
`)
} catch (e) {
  console.log('✓ Caught type error:', (e as Error).message)
}

// Example 5: Complex nested function calls
console.log('\nExample 5: Nested Function Calls')
typecheckTypeScript(`
function f(x: number): number { return x }
function g(x: number): number { return x }
function h(x: number): number { return x }

h(g(f(1)))
`)
