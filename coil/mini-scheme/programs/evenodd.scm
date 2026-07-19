(define (evn n) (if (= n 0) 1 (od (- n 1))))
(define (od n) (if (= n 0) 0 (evn (- n 1))))
(define (bench k last) (if (= k 0) last (bench (- k 1) (evn 10000))))
(display (bench 40000 0)) (newline)
