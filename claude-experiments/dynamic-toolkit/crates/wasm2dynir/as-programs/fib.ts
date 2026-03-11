export function fib(n: i32): i32 {
  let a: i32 = 0;
  let b: i32 = 1;
  for (let i: i32 = 0; i < n; i++) {
    let tmp = a + b;
    a = b;
    b = tmp;
  }
  return a;
}
