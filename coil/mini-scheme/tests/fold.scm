(define fold (lambda (f acc xs) (if (null? xs) acc (fold f (f acc (car xs)) (cdr xs)))))
(define range (lambda (a b) (if (< a b) (cons a (range (+ a 1) b)) (quote ()))))
(define add (lambda (a b) (+ a b)))
(fold add 0 (range 1 101))
