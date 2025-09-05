// Test variable scoping and context handling

function outerScope() {
  let x = 42;
  let y = "hello";
  
  if (x > 0) {
    let z = x + 10;        // z should be number (42 + 10)
    let message = y + z;   // message should be string ("hello52")
    return message;
  }
  
  return y;
}

function parameterScope(a: number, b: string) {
  let c = a * 2;         // c should be number
  let d = b + c;         // d should be string  
  return d;
}

function undefinedVariable() {
  let x = unknownVar;    // Should error: unknownVar not defined
  return x;
}

function shadowingTest(x: number) {
  let result = x * 2;    // result should be number
  
  if (x > 0) {
    let result = "positive";  // Shadows outer result, should be string
    return result;            // Should return string
  }
  
  return result;         // Should return number - this creates inconsistent returns!
}