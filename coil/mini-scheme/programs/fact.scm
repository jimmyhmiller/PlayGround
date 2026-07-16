(define (fact n) (if (< n 2) 1 (* n (fact (- n 1)))))
(display (fact 12)) (newline)
