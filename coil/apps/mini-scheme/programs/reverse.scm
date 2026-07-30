(define (range n) (if (= n 0) (quote ()) (cons n (range (- n 1)))))
(define (rev xs acc) (if (null? xs) acc (rev (cdr xs) (cons (car xs) acc))))
(define (bench k last) (if (= k 0) last (bench (- k 1) (rev (range 20) (quote ())))))
(display (bench 1000000 (quote ()))) (newline)
