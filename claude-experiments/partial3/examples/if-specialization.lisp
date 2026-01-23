; If-specialization example
; Run with: --dynamic x
; The condition is static (true), so only the then-branch survives
(let flag true
  (if flag
    (+ x 10)
    (+ x 20)))
