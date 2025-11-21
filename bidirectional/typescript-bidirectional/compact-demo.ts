import { parse } from './compact-ohm-parser'

console.log('=== Compact Ohm Parser ===\n')

// Simple examples
console.log('1. Number:', parse('42').type)
console.log('2. Let:', parse('let x = 10; x').type)
console.log('3. Type annotation:', parse('let x: number = 42; x').type)
console.log('4. Multiple lets:', parse('let x = 5; let y = 10; y').type)

// Type error
try {
  parse('let x: string = 42; x')
} catch (e) {
  console.log('5. Type error caught! ✓')
}

console.log('\n✨ That\'s it! Grammar + mapper in ~50 lines')
