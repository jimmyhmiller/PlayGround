#!/bin/bash

echo "=== JIT Compiler Benchmark ==="
echo ""

FIB_CODE='(def fib (fn [n] (if (< n 2) n (+ (fib (- n 1)) (fib (- n 2))))))'

echo "fib(30) - 3 runs:"
for i in 1 2 3; do
    /usr/bin/time -p sh -c "echo '$FIB_CODE
(fib 30)' | ./target/release/quick-clojure-poc 2>/dev/null >/dev/null" 2>&1 | grep real
done

echo ""
echo "fib(35) - 3 runs:"
for i in 1 2 3; do
    /usr/bin/time -p sh -c "echo '$FIB_CODE
(fib 35)' | ./target/release/quick-clojure-poc 2>/dev/null >/dev/null" 2>&1 | grep real
done

echo ""
echo "=== Clojure Benchmark ==="
echo ""

CLJ_BENCH='(def fib (fn [n] (if (< n 2) n (+ (fib (- n 1)) (fib (- n 2))))))
(defn run-bench [n]
  (let [start (System/nanoTime)]
    (fib n)
    (/ (- (System/nanoTime) start) 1000000.0)))
(println "Warming up...")
(dotimes [_ 3] (fib 30))
(println "fib(30):" (run-bench 30) "ms")
(println "fib(30):" (run-bench 30) "ms")
(println "fib(30):" (run-bench 30) "ms")
(println "fib(35):" (run-bench 35) "ms")
(println "fib(35):" (run-bench 35) "ms")
(println "fib(35):" (run-bench 35) "ms")'

echo "$CLJ_BENCH" | clj -M -
