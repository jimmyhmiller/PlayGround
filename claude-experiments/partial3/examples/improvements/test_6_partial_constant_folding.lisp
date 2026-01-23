; Test case for Improvement #6: Arithmetic on Dynamic with Static Pattern
; Expressions mixing static and dynamic values should apply partial constant folding.
;
; Run with: cargo run -- examples/improvements/test_6_partial_constant_folding.lisp --dynamic input

; Test 1: (+ x 0) should fold to x
(+ input 0)
; Expected: input

; Test 2: (* x 1) should fold to x
(* input 1)
; Expected: input

; Test 3: (* x 0) should fold to 0
(* input 0)
; Expected: 0

; Test 4: Nested constant folding
; (+ (+ x 1) 2) should fold to (+ x 3)
(+ (+ input 1) 2)
; Expected: (+ input 3)

; Test 5: Bitwise AND reassociation
; (& (& x 255) 15) should fold to (& x 15)
(& (& input 255) 15)
; Expected: (& input 15)

; Test 6: Idempotent mask operations
; (& x 255) when x is already (& y 255) shouldn't change
; This is harder to detect but common in byte manipulation
(let x (& input 255)
  (& x 255))
; Expected: (& input 255) - single mask, not nested
