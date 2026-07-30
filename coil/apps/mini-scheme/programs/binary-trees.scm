(define (make d) (if (= d 0) (cons 0 0) (cons (make (- d 1)) (make (- d 1)))))
(define (check t) (if (pair? (car t)) (+ 1 (+ (check (car t)) (check (cdr t)))) 1))
(define (bench k last) (if (= k 0) last (bench (- k 1) (check (make 14)))))
(display (bench 800 0)) (newline)
