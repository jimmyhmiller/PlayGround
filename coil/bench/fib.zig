// Naive recursive Fibonacci — matches fib.c / fib.coil.
const c = @cImport(@cInclude("stdio.h"));
fn fib(n: i64) i64 {
    return if (n < 2) n else fib(n - 1) + fib(n - 2);
}
export fn main() c_int {
    _ = c.printf("%ld\n", fib(40));
    return 0;
}
