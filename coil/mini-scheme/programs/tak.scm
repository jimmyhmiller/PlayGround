(define (tak x y z) (if (< y x) (tak (tak (- x 1) y z) (tak (- y 1) z x) (tak (- z 1) x y)) z))
(define (bench k last) (if (= k 0) last (bench (- k 1) (tak 18 12 6))))
(display (bench 1500 0)) (newline)
