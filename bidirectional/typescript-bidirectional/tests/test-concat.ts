// Test string/number concatenation edge cases

function testConcat(): string {
  let a = "hello";
  let b = 42;
  let c = 3.14;
  
  // All of these should work and return string
  let result1 = a + b;        // string + number = string
  let result2 = b + a;        // number + string = string  
  let result3 = a + b + c;    // string + number + number = string
  let result4 = b + c + a;    // number + number + string = ???
  
  return result4;
}

function testArithmetic(): number {
  let x = 10;
  let y = 20;
  
  // These should all be number
  let add = x + y;
  let sub = x - y; 
  let mul = x * y;
  let div = x / y;
  
  return add + sub + mul + div;
}

function badConcat(): string {
  let x = 5;
  let y = 10;
  
  // This should be number (5 + 10), but we're returning it as string
  return x + y;  // Should error: number to string
}