// Test differences between checking and inference modes

// CHECKING MODE: Explicit return type guides type checking
function explicitString(x: number): string {
  if (x > 0) {
    return "positive: " + x;  // Checked against string
  }
  return "zero or negative";   // Checked against string
}

// INFERENCE MODE: Type is inferred from return statements  
function inferredType(x: number) {
  if (x > 0) {
    return "positive: " + x;  // First return determines type as string
  }
  return "zero or negative";   // Must match first return type
}

// This should fail in CHECKING mode
function badExplicit(x: number): string {
  if (x > 0) {
    return x;  // Error: number doesn't match declared string
  }
  return "negative";
}

// This should fail in INFERENCE mode  
function badInferred(x: number) {
  if (x > 0) {
    return "positive";  // First return is string
  }
  return 42;            // Error: number doesn't match inferred string
}

// Test that function calls use the right mode
function takesString(s: string): string {
  return s + "!";
}

function testCalls() {
  // Arguments are CHECKED against parameter types
  let result1 = takesString("hello");     // OK: string matches string
  let result2 = takesString(123);         // Error: number doesn't match string parameter
}