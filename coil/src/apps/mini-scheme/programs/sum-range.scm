(define (range n) (if (= n 0) (quote ()) (cons n (range (- n 1)))))
(define (sum xs) (if (null? xs) 0 (+ (car xs) (sum (cdr xs)))))
(define (bench k last) (if (= k 0) last (bench (- k 1) (sum (range 1000)))))
(display (bench 12000 0)) (newline)
