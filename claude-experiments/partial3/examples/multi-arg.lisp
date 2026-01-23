; Multi-argument partial application
; Run with: --dynamic input
; a and c are known, b (input) is dynamic
(let compute (fn (a b c) (+ (* a b) c))
  (call compute 3 input 7))
