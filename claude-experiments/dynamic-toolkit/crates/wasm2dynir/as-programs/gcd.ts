export function gcd(a: i32, b: i32): i32 {
  while (b != 0) {
    let t = b;
    b = a % b;
    a = t;
  }
  return a;
}
