; Test case for Improvement #1: Object Property Tracking
; Currently, once an object has a property set, subsequent reads don't
; resolve to the known value even though we track it.
;
; Expected: After (prop-set! obj "x" 5), (prop obj "x") should return 5 statically
; Current behavior: The prop-set! works but reading may not optimize further

; Run with: cargo run -- examples/improvements/test_1_object_property_tracking.lisp --dynamic input

; Test 1: Basic property read after set should be static
(let obj (object)
  (begin
    (prop-set! obj "x" 5)
    ; This should be optimized to just 5, not (prop obj "x")
    (+ (prop obj "x") input)))

; Expected residual: (+ 5 input)
; If object property tracking is not working: (let obj ... (+ (prop obj "x") input))
