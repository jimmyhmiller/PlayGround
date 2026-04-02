// Node.js benchmark for comparison with dynlower JIT
// Run: node benches/node_comparison.js

function bench(name, fn, warmup = 1000, iterations = 10000) {
    // Warmup
    for (let i = 0; i < warmup; i++) fn();

    const start = process.hrtime.bigint();
    for (let i = 0; i < iterations; i++) fn();
    const elapsed = Number(process.hrtime.bigint() - start);
    const per_iter = elapsed / iterations;
    console.log(`${name}: ${per_iter.toFixed(1)} ns/iter (${iterations} iterations)`);
}

// Fibonacci (iterative)
function fib(n) {
    let a = 0, b = 1;
    for (let i = 0; i < n; i++) {
        const tmp = a + b;
        a = b;
        b = tmp;
    }
    return a;
}

// Factorial (iterative)
function factorial(n) {
    let result = 1;
    while (n > 1) {
        result *= n;
        n--;
    }
    return result;
}

// Sum 0..n
function sum(n) {
    let acc = 0;
    for (let i = 0; i < n; i++) {
        acc += i;
    }
    return acc;
}

// Nested loop
function nested_loop(n) {
    let acc = 0;
    for (let i = 0; i < n; i++) {
        for (let j = 0; j < n; j++) {
            acc += i * j;
        }
    }
    return acc;
}

console.log("=== Node.js v" + process.version + " ===");
bench("fib(30)", () => fib(30));
bench("factorial(20)", () => factorial(20));
bench("sum(10000)", () => sum(10000), 100, 1000);
bench("nested_loop(100)", () => nested_loop(100), 100, 1000);
