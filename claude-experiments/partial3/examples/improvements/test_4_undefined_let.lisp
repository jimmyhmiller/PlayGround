; Test case for Improvement #4: Let-binding of undefined Elimination
; Let bindings initialized to undefined that are immediately set! should be
; eliminated or hoisted.
;
; Run with: cargo run -- examples/improvements/test_4_undefined_let.lisp --dynamic input

; Test 1: Variable bound to undefined, then immediately set
; This pattern is common in generated JavaScript
(let x undefined
  (begin
    (set! x 42)
    (+ x input)))

; Expected residual: (let x 42 (+ x input)) or (+ 42 input)
; Not: (let x undefined (begin (set! x 42) (+ x input)))

; Test 2: Multiple undefined bindings
(let a undefined
  (let b undefined
    (begin
      (set! a 1)
      (set! b 2)
      (+ (+ a b) input))))

; Expected: Should simplify to (+ 3 input) or at minimum eliminate the undefineds
