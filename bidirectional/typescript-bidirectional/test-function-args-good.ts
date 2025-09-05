// Test function argument type checking - good version

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
  
  // Nested calls - convert number to string first
  let numberResult = takesNumber(5);
  let stringified = "Result: " + numberResult;
  let finalResult = takesString(stringified);
}