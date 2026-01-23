; Partial evaluation example
; Run with: --dynamic x
; The multiplier is known, but x is not
(let multiplier 3
  (* x multiplier))
