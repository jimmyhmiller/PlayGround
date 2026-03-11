// Count primes up to n using trial division
export function count_primes(n: i32): i32 {
  let count: i32 = 0;
  for (let i: i32 = 2; i <= n; i++) {
    if (is_prime(i)) {
      count++;
    }
  }
  return count;
}

function is_prime(n: i32): bool {
  if (n < 2) return false;
  for (let i: i32 = 2; i * i <= n; i++) {
    if (n % i == 0) return false;
  }
  return true;
}
