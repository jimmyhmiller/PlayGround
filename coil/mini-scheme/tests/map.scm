(define map (lambda (f xs) (if (null? xs) (quote ()) (cons (f (car xs)) (map f (cdr xs))))))
(define range (lambda (a b) (if (< a b) (cons a (range (+ a 1) b)) (quote ()))))
(define sq (lambda (x) (* x x)))
(map sq (range 1 8))
