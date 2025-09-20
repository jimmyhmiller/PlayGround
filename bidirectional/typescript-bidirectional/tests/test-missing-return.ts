// Test functions missing required return statements

function declaredNumberButNoReturn(): number {
  let x = 5;
  let y = x * 2;
  // No return statement but declared as returning number - should error
}