; Test that variables set in one case are NOT incorrectly inlined in other cases
; when the discriminant is dynamic
;
; Run with: cargo run -- examples/improvements/test_switch_crosscase.lisp --dynamic state

; v38 starts as undefined, but case 13 sets it
; case 11 should NOT inline undefined for v38
(let v38 undefined
  (switch state
    (case 11
      ; v38 should remain as v38, NOT be inlined as undefined
      (new DataView v38))
    (case 13
      (set! v38 (new ArrayBuffer 8)))))

; Expected residual should keep v38 as a variable reference in case 11, like:
; (switch state
;   (case 11 (new DataView v38))
;   (case 13 (set! v38 (new ArrayBuffer 8))))
;
; NOT: (case 11 (new DataView undefined))
