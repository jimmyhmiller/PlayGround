#!/usr/bin/env bash
# Benchmark mini-scheme vs Chez/Petite. Needs hyperfine + chez + petite on PATH.
# Run from mini-scheme/.
set -e
../coil build scheme.coil -o /tmp/mini-scheme >/dev/null
printf '1\n'                                                            > /tmp/ms_triv.scm
printf '(display 1)(newline)\n'                                        > /tmp/cz_triv.scm
printf '(define fib (lambda (n) (if (< n 2) n (+ (fib (- n 1)) (fib (- n 2))))))\n(fib 30)\n' > /tmp/ms_fib.scm
printf '(define fib (lambda (n) (if (< n 2) n (+ (fib (- n 1)) (fib (- n 2))))))\n(display (fib 30))(newline)\n' > /tmp/cz_fib.scm
echo "=== startup (trivial program) ==="
hyperfine -w3 -r20 "/tmp/mini-scheme < /tmp/ms_triv.scm" "chez -q --script /tmp/cz_triv.scm" "petite -q --script /tmp/cz_triv.scm"
echo "=== fib(30) — 2.7M calls, 47M allocations in mini-scheme ==="
hyperfine -w2 -r12 "/tmp/mini-scheme < /tmp/ms_fib.scm" "chez -q --script /tmp/cz_fib.scm" "petite -q --script /tmp/cz_fib.scm"
echo "=== mini-scheme GC stats on fib(30) ==="
/tmp/mini-scheme < /tmp/ms_fib.scm 2>&1 >/dev/null
