(define (append2 a b) (if (null? a) b (cons (car a) (append2 (cdr a) b))))
(define (range n) (if (= n 0) (quote ()) (cons n (range (- n 1)))))
(define (len xs) (if (null? xs) 0 (+ 1 (len (cdr xs)))))
(display (len (append2 (range 30) (range 20)))) (newline)
