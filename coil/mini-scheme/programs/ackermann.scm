(define (ack m n) (if (= m 0) (+ n 1) (if (= n 0) (ack (- m 1) 1) (ack (- m 1) (ack m (- n 1))))))
(define (bench k last) (if (= k 0) last (bench (- k 1) (ack 3 6))))
(display (bench 300 0)) (newline)
