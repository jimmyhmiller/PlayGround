// Example demonstrating Rust-like numeric types with type inference
// Run this file directly to see how the type inference works

console.log("=== Rust-like Numeric Types with Type Inference ===\n");

console.log("The bidirectional type checker now supports multiple numeric types:");
console.log("- Unsigned: u8, u16, u32, u64, u128, usize");
console.log("- Signed: i8, i16, i32, i64, i128, isize");
console.log("- Float: f32, f64 (prepared for future implementation)\n");

console.log("Key Features:");
console.log("1. When you have a variable with a specific numeric type (e.g., x: usize)");
console.log("2. And you do arithmetic with a literal (e.g., x + 1)");
console.log("3. The literal is automatically inferred to have the same type as the variable");
console.log("4. But only if the literal can fit in that type!\n");

console.log("Examples from the test suite:\n");

console.log("✓ x: usize, then x + 1");
console.log("  The literal 1 is inferred as usize\n");

console.log("✓ y: u8, then y + 100");
console.log("  100 fits in u8 (max 255), so it works\n");

console.log("✗ z: u8, then z + 300");
console.log("  300 doesn't fit in u8, so this fails with an error\n");

console.log("✓ count: u32, then 2 * count");
console.log("  The literal 2 is inferred as u32\n");

console.log("✓ val: i8, then val - 10");
console.log("  The literal 10 is inferred as i8 (fits in range -128 to 127)\n");

console.log("The type checker also works with complex nested expressions:");
console.log("✓ x: u32, then (x * 2) + 5");
console.log("  Both 2 and 5 are inferred as u32\n");

console.log("This feature makes the type system more ergonomic, similar to Rust,");
console.log("where you don't need to annotate every numeric literal when the type");
console.log("can be inferred from context!");

console.log("\nTo run the full test suite, execute:");
console.log("  node bidirectional2.js");