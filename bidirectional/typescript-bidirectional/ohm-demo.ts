import { parseAndTypeCheck } from './ohm-to-minimal-mapper'

console.log('=== Ohm-based Parser Demo ===\n')
console.log('Direct parsing from concrete syntax to minimal AST!\n')
console.log('='.repeat(60))

// Example 1: Simple number
console.log('\n📝 Example 1: Simple number')
console.log('-'.repeat(60))
parseAndTypeCheck('42')

// Example 2: Let binding
console.log('\n📝 Example 2: Let binding')
console.log('-'.repeat(60))
parseAndTypeCheck(`
let x = 10;
x
`)

// Example 3: Let with type annotation
console.log('\n📝 Example 3: Let with type annotation')
console.log('-'.repeat(60))
parseAndTypeCheck(`
let x: number = 42;
x
`)

// Example 4: Simple variable reference
console.log('\n📝 Example 4: Variable reference')
console.log('-'.repeat(60))
parseAndTypeCheck(`
let x = 42;
let y = x;
y
`)

// Example 5: Multiple let bindings
console.log('\n📝 Example 5: Multiple let bindings')
console.log('-'.repeat(60))
parseAndTypeCheck(`
let x = 5;
let y = 10;
let z = 20;
z
`)

// Example 6: String type
console.log('\n📝 Example 6: String type')
console.log('-'.repeat(60))
parseAndTypeCheck(`
let message: string = "hello";
message
`)

// Example 7: Type error detection
console.log('\n📝 Example 7: Type error detection')
console.log('-'.repeat(60))
try {
  parseAndTypeCheck(`
let x: string = 42;
x
`)
} catch (e) {
  console.log('✓ Caught type error:', (e as Error).message)
}

console.log('\n' + '='.repeat(60))
console.log('🎉 All examples completed!')
console.log('='.repeat(60))
