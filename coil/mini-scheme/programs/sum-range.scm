(define (range n) (if (= n 0) (quote ()) (cons n (range (- n 1)))))
(define (sum xs) (if (null? xs) 0 (+ (car xs) (sum (cdr xs)))))
(display (sum (range 1000))) (newline)
