(define (fact n) (if (< n 2) 1 (* n (fact (- n 1)))))
(define (bench k last) (if (= k 0) last (bench (- k 1) (fact 12))))
(display (bench 10000000 0)) (newline)
