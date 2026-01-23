; While loop with switch - should unroll through static cases until hitting dynamic
;
; Expected behavior:
; - state starts at 0
; - Loop iteration 1: state=0 -> case 0 -> state=1
; - Loop iteration 2: state=1 -> case 1 -> state=2
; - Loop iteration 3: state=2 -> case 2 -> state=3
; - Loop iteration 4: state=3 -> case 3 -> state=input (DYNAMIC!)
; - Now state is dynamic, emit residual while loop
;
; Run with: cargo run -- examples/improvements/test_while_switch_partial_unroll.lisp --dynamic input

(let state 0
  (let result 0
    (begin
      (while (>= state 0)
        (switch state
          (case 0
            (set! result (+ result 100))
            (set! state 1))
          (case 1
            (set! result (+ result 200))
            (set! state 2))
          (case 2
            (set! result (+ result 300))
            (set! state 3))
          (case 3
            ; This case is dynamic - state becomes unknown
            (set! result (+ result input))
            (set! state input))
          (case 4
            (set! result (+ result 400))
            (set! state -1))
          (default
            (set! state -1))))
      result)))

; Expected: We should see result accumulate 100+200+300=600 statically,
; then a residual while loop for the remaining dynamic iterations.
;
; Something like:
; (let state input
;   (let result (+ 600 input)
;     (begin
;       (while (>= state 0)
;         (switch state ...))
;       result)))
