(define (evn n) (if (= n 0) 1 (od (- n 1))))
(define (od n) (if (= n 0) 0 (evn (- n 1))))
(display (evn 100000)) (newline)
