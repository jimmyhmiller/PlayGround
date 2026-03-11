// Fast exponentiation by squaring
export function power(base: i64, exp: i32): i64 {
  let result: i64 = 1;
  let b: i64 = base;
  let e: i32 = exp;
  while (e > 0) {
    if (e & 1) {
      result = result * b;
    }
    b = b * b;
    e = e >> 1;
  }
  return result;
}
