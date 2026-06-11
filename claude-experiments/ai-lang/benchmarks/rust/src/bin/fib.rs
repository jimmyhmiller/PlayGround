// Naive recursive Fibonacci — mirrors benchmarks/ail/fib.ail.

fn fib(n: i64) -> i64 {
    if n < 2 { n } else { fib(n - 1) + fib(n - 2) }
}

fn main() {
    let n = 32;
    let t0 = std::time::Instant::now();
    let r = fib(n);
    let ms = t0.elapsed().as_millis();
    println!("RESULT fib {} ms checksum={}", ms, r);
}
