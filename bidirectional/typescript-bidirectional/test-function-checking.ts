import { typecheckTypeScript } from './ts-to-minimal-mapper'

console.log('\n=== Testing Function Body Checking ===\n')

// Test 1: Correct function - should pass
try {
  typecheckTypeScript(`
function id(x: number): number { return x }
id(5)
`.trim())
  console.log('\n✓ Correct function passed')
} catch (e) {
  console.log('\n✗ Correct function failed:', (e as Error).message)
}

// Test 2: Function returning wrong type - should fail
try {
  typecheckTypeScript(`
function bad(x: number): number { return "oops" }
bad(5)
`.trim())
  console.log('\n✗ Bad function passed (unexpected!)')
} catch (e) {
  console.log('\n✓ Bad function caught:', (e as Error).message)
}

// Test 3: Function with multiple statements
try {
  typecheckTypeScript(`
function compute(x: number): number {
  let y = 10
  return x
}
compute(5)
`.trim())
  console.log('\n✓ Multi-statement function passed')
} catch (e) {
  console.log('\n✗ Multi-statement function failed:', (e as Error).message)
}
