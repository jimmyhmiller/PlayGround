(define (gcd a b) (if (= a b) a (if (< a b) (gcd a (- b a)) (gcd (- a b) b))))
(display (gcd 1071 462)) (newline)
