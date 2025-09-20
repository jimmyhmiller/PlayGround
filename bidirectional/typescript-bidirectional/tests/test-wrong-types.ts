// Let's try to find edge cases where types get confused

function typeConfusion() {
  let x = 5;           // x is number
  let y = "hello";     // y is string
  
  if (x > 0) {
    let z = y + x;     // z should be string ("hello5")
    let w = x + x;     // w should be number (10)  
    let bad = z + w;   // This should work: string + number = string
    return bad;        // Should be string
  }
  
  return y;            // Should be string - both paths return string
}

function arithmeticVsConcat() {
  let a = 1;
  let b = 2;  
  let c = "3";
  
  let math = a + b;      // number + number = number (3)
  let concat1 = c + a;   // string + number = string ("31")
  let concat2 = a + c;   // number + string = string ("13")
  
  // Now the tricky part - what about complex expressions?
  let complex = a + b + c;  // (1 + 2) + "3" = 3 + "3" = "33" (string)
  
  return complex;
}