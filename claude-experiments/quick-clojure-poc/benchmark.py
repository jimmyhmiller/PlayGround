#!/usr/bin/env python3
import subprocess
import time

print("=== Recursive Fibonacci Benchmark ===")
print()

for n in [30, 33, 35]:
    # Include warmup - call fib(20) first to warm up JIT, then measure fib(n)
    FIB_CODE = f'''(def fib (fn [n] (if (< n 2) n (+ (fib (- n 1)) (fib (- n 2))))))
(fib 20)
(fib {n})'''

    print(f"fib({n}):")

    # JIT Compiler - run 5 times and take best 3
    times = []
    for i in range(5):
        start = time.time()
        result = subprocess.run(
            ['./target/release/quick-clojure-poc'],
            input=FIB_CODE,
            capture_output=True,
            text=True,
            timeout=120,
            env={**subprocess.os.environ, 'RUST_LOG': 'off'}
        )
        end = time.time()
        elapsed = (end - start) * 1000
        times.append(elapsed)
    # Take best 3 of 5 to reduce variance from startup
    times.sort()
    best_3 = times[:3]
    avg = sum(best_3) / len(best_3)
    print(f"  JIT:     {avg:.1f}ms (best 3 of 5)")

    # Clojure
    clj_code = f'''
(def fib (fn [n] (if (< n 2) n (+ (fib (- n 1)) (fib (- n 2))))))
(dotimes [_ 2] (fib 25))
(let [times (for [_ (range 3)]
              (let [start (System/nanoTime)]
                (fib {n})
                (/ (- (System/nanoTime) start) 1000000.0)))]
  (println (format "  Clojure: %.1fms (avg of 3)" (/ (reduce + times) 3.0))))
'''
    result = subprocess.run(['clj', '-M', '-e', clj_code], capture_output=True, text=True, timeout=120)
    print(result.stdout.strip())
    print()
