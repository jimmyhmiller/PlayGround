// Test void functions - good cases

function correctVoid(): void {
  let x = 5;
  let y = x * 2;
  // No return statement - should be void
}

function explicitVoidReturn(): void {
  let x = 10;
  return;  // Explicit void return - should be fine
}

function shouldBeVoid() {
  // No explicit return type, no return statement - should infer void
  let temp = "hello";
}