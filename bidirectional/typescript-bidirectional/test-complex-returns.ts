// Test mixed return types in complex control flow

function complexReturns(x: number): string {
  if (x > 10) {
    if (x > 20) {
      return "big: " + x;
    }
    return "medium: " + x;
  }
  
  if (x < 0) {
    return 42;  // Should error: number instead of string
  }
  
  return "small: " + x;
}

function nestedIfReturns(flag: boolean): number {
  if (flag) {
    if (flag) {
      return 1;
    }
    return "two";  // Should error: string instead of number  
  }
  return 3;
}