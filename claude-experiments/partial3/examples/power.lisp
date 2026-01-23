; A more complex example: manual power function unrolled
; Run with: --dynamic x
; This computes x^4 with the exponent (4) being static
(let square (fn (n) (* n n))
  (let x2 (call square x)
    (call square x2)))
