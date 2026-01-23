; Simpler test: while loop that becomes dynamic after 3 iterations
; Run with: cargo run -- examples/improvements/test_simple_partial_unroll.lisp --dynamic input

(let x 0
  (let result 0
    (begin
      (while (< x 5)
        (if (== x 3)
          ; At x=3, result depends on input - body becomes dynamic
          (begin
            (set! result (+ result input))
            (set! x 10))
          ; For x=0,1,2 - fully static
          (begin
            (set! result (+ result 100))
            (set! x (+ x 1)))))
      result)))

; Expected unrolling:
; x=0: result=100, x=1
; x=1: result=200, x=2
; x=2: result=300, x=3
; x=3: result=300+input, x=10 (dynamic!)
;
; Residual should be:
; (begin
;   (set! x 10)
;   (set! result (+ 300 input))
;   (while (< x 5) ...))  ; but x=10, so loop exits immediately
;
; Or just: (+ 300 input) if we're smart about it
