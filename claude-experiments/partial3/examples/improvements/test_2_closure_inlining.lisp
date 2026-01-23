; Test case for Improvement #2: Closure Inlining for Non-Mutating Closures
; Closures that don't mutate captured state should be inlined at call sites,
; even if they read from mutable variables.
;
; Run with: cargo run -- examples/improvements/test_2_closure_inlining.lisp --dynamic input

; Test 1: Closure that only reads captured state (doesn't mutate it)
; The closure f captures 'x' but never mutates it
(let x 10
  (let f (fn (y) (+ y x))
    ; This call should be inlined to (+ input 10)
    (call f input)))

; Expected residual: (+ input 10)
; If closure not inlined: (let f (fn (y) (+ y 10)) (call f input))
