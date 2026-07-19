(define (fib n) (if (< n 2) n (+ (fib (- n 1)) (fib (- n 2)))))
(define (bench k last) (if (= k 0) last (bench (- k 1) (fib 30))))
(display (bench 48 0)) (newline)
