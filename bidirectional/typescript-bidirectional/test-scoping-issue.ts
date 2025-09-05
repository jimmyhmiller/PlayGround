// This reveals a scoping bug in our reduced type checker

function scopeBug(x: number) {
  let result = x * 2;    // result is number
  
  if (x > 0) {
    let result = "positive";  // This should shadow the outer result
    // But our type checker probably doesn't handle block scoping!
  }
  
  return result;         // Should still be number from outer scope
}