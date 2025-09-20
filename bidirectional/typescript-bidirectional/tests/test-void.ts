// Test void function validation

function correctVoid(): void {
  let x = 5;
  let y = x * 2;
  // No return statement - should be void
}

function explicitVoidReturn(): void {
  let x = 10;
  return;  // Explicit void return - should be fine
}

function wrongVoid(): void {
  let result = 42;
  return result;  // Should error: returning number from void function
}

function shouldBeVoid() {
  // No explicit return type, no return statement
  let temp = "hello";
}

function wronglyDeclaredVoid(): number {
  // Declared as number but has no return - should error
  let x = 5;
}