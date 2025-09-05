// Function with explicit return type (uses checking mode)
function explicitReturn(x: number): string {
  return "Result: " + x;
}

// Function without explicit return type (uses inference mode)
function inferredReturn(x: number) {
  return "Result: " + x;
}

// Function call (uses bidirectional checking for arguments)
function test() {
  let result1 = explicitReturn(42);
  let result2 = inferredReturn(42);
}