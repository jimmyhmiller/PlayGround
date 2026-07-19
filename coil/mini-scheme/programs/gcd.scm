(define (gcd a b) (if (= a b) a (if (< a b) (gcd a (- b a)) (gcd (- a b) b))))
(define (bench k last) (if (= k 0) last (bench (- k 1) (gcd 1071 462))))
(display (bench 12000000 0)) (newline)
