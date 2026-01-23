; Closure specialization example
; Run with: --dynamic input
; make-adder creates a closure, add5 is specialized, and we call it with dynamic input
(let make-adder (fn (n) (fn (x) (+ x n)))
  (let add5 (call make-adder 5)
    (call add5 input)))
