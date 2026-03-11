// Count steps to reach 1 in the Collatz sequence
export function collatz(n: i32): i32 {
  let steps: i32 = 0;
  while (n != 1) {
    if (n % 2 == 0) {
      n = n / 2;
    } else {
      n = 3 * n + 1;
    }
    steps++;
  }
  return steps;
}
