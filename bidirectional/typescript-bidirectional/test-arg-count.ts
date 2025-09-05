// Test argument count validation

function twoParams(x: number, y: string): string {
  return y + x;
}

function testArguments() {
  // Correct calls
  let good1 = twoParams(42, "hello");
  
  // Wrong argument counts - these should fail
  let bad1 = twoParams(42);              // Too few arguments
  let bad2 = twoParams(42, "hello", 99); // Too many arguments  
  let bad3 = twoParams();                // No arguments
  
  return good1;
}