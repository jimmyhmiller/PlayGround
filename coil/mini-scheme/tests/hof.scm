(define compose (lambda (f g) (lambda (x) (f (g x)))))
(define inc (lambda (x) (+ x 1)))
(define dbl (lambda (x) (* x 2)))
((compose inc dbl) 20)
