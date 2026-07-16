(define (range n) (if (= n 0) (quote ()) (cons n (range (- n 1)))))
(define (rev xs acc) (if (null? xs) acc (rev (cdr xs) (cons (car xs) acc))))
(display (rev (range 6) (quote ()))) (newline)
