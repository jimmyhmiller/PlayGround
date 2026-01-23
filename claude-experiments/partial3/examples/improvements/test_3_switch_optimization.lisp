; Test case for Improvement #3: State Machine / Switch Optimization
; When a switch discriminant is the result of a bitwise operation with static mask,
; we should be able to compute the initial case and potentially specialize.
;
; Run with: cargo run -- examples/improvements/test_3_switch_optimization.lisp --dynamic input

; Test 1: Static discriminant with bitwise AND should select correct case
; (& 1605 15) = 5, so case 5 should be selected
(let state 1605
  (switch (& state 15)
    (case 0 100)
    (case 5 555)
    (case 10 1000)
    (default 999)))

; Expected: 555 (fully static, no switch needed)

; Test 2: Partially static - mask known but input varies
; When we can't fully evaluate, at least fold constants
(switch (& (+ input 128) 15)
  (case 0 100)
  (case 5 555)
  (case 10 1000)
  (default 999))

; This should keep the switch but with (& (+ input 128) 15) as discriminant
