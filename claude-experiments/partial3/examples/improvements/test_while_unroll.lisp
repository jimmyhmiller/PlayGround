; Test that while loops fully unroll when all values are static
(let x 0
  (begin
    (while (< x 5)
      (set! x (+ x 1)))
    x))
; Expected: 5 (fully evaluated, no residual)
