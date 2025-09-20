// Test concatenation order - this reveals an issue with our type checker

function testOrder(): string {
  let a = "hello";
  let b = 42;
  let c = 10;
  
  // b + c = number (42 + 10 = 52)
  // then 52 + a should be string ("52hello")  
  let result = b + c + a;
  
  return result;
}