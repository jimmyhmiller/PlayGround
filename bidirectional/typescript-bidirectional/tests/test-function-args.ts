// Test function argument type checking

function takesNumber(x: number): number {
  return x * 2;
}

function takesString(s: string): string {
  return s + "!";
}

function testCalls() {
  // These should work
  let good1 = takesNumber(42);
  let good2 = takesString("hello");
  
  // These should fail
  let bad1 = takesNumber("string");  // Should error: string to number
  let bad2 = takesString(123);       // Should error: number to string
  
  // Nested calls
  let nested = takesString(takesNumber(5)); // Should error: number to string
}