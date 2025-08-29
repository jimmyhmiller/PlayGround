import { processProgram, formatType, runFail } from './index.js';

console.log('🌊 Control Flow with If Statements Demo\n');

// Example 1: Consistent types across branches
console.log('✅ Example 1: Consistent types across branches');
const example1 = `
  let score
  if (true) {
    score = 95
  } else {
    score = 87
  }
  score + 5
`;
console.log(example1);
const result1 = processProgram(example1);
console.log(`✓ score type: ${formatType(result1.context.score)}\n`);

// Example 2: Complex expressions in branches  
console.log('✅ Example 2: Complex expressions in branches');
const example2 = `
  let total
  let tax = 0.08
  let basePrice = 100
  if (basePrice > 50) {
    total = basePrice * (1 + tax)
  } else {
    total = basePrice + 5
  }
`;
console.log(example2);
const result2 = processProgram(example2);
console.log(`✓ total type: ${formatType(result2.context.total)}\n`);

// Example 3: Multiple variables in branches
console.log('✅ Example 3: Multiple variables in branches');
const example3 = `
  let width
  let height  
  let area
  if (true) {
    width = 10
    height = 20
  } else {
    width = 15
    height = 25
  }
  area = width * height
`;
console.log(example3);
const result3 = processProgram(example3);
console.log(`✓ width: ${formatType(result3.context.width)}`);
console.log(`✓ height: ${formatType(result3.context.height)}`);
console.log(`✓ area: ${formatType(result3.context.area)}\n`);

// Example 4: Nested if statements
console.log('✅ Example 4: Nested if statements');
const example4 = `
  let category
  let priority
  let age = 25
  if (age > 18) {
    category = "adult"
    if (age > 65) {
      priority = 1
    } else {
      priority = 2  
    }
  } else {
    category = "minor"
    priority = 3
  }
`;
console.log(example4);
const result4 = processProgram(example4);
console.log(`✓ category: ${formatType(result4.context.category)}`);
console.log(`✓ priority: ${formatType(result4.context.priority)}\n`);

console.log('🚫 Error Cases (properly caught):');

// Error 1: Different types
console.log('1. Different types in branches:');
try {
  processProgram(`
    let x
    if (true) {
      x = 42
    } else {
      x = "hello"
    }
  `);
  console.log('✗ Should have failed!');
} catch (e) {
  console.log(`✓ ${e.message}\n`);
}

// Error 2: Assignment only in then branch
console.log('2. Assignment only in then branch:');
try {
  processProgram(`
    let x
    if (true) {
      x = 42
    }
    x + 1
  `);
  console.log('✗ Should have failed!');
} catch (e) {
  console.log(`✓ ${e.message}\n`);
}

// Error 3: Unbalanced assignments
console.log('3. Unbalanced assignments:');
try {
  processProgram(`
    let x
    if (true) {
      // no assignment
    } else {
      x = 42
    }
  `);
  console.log('✗ Should have failed!');
} catch (e) {
  console.log(`✓ ${e.message}\n`);
}

console.log('🎯 Key Benefits:');
console.log('• Variables can be declared without initialization');
console.log('• Types inferred from assignments in control flow');
console.log('• Type consistency enforced across all branches');
console.log('• Variables must be assigned in ALL paths to be usable');
console.log('• Prevents runtime undefined variable errors');
console.log('• Works with nested if statements');
console.log('• Integrates seamlessly with function type inference');