; GAP: scf.while operation
; scf.while is a more general loop construct with before/after regions
; This file tests scf.while support

(require-dialect [func :as f] [arith :as a] [scf :as s])

(module
  (do
    ; Test 1: Basic scf.while loop
    ; In MLIR:
    ; %result = scf.while (%arg = %init) : (i64) -> (i64) {
    ;   %cond = arith.cmpi slt, %arg, %bound : i64
    ;   scf.condition(%cond) %arg : i64
    ; } do {
    ; ^bb0(%arg: i64):
    ;   %next = arith.addi %arg, %c1 : i64
    ;   scf.yield %next : i64
    ; }
    (f/func {:sym_name "count_while"
             :function_type (-> [i64 i64] [i64])
             :llvm.emit_c_interface true}
      (region
        (block [(: init i64) (: bound i64)]
          (def c1 (: 1 i64))
          ; scf.while has TWO regions: before (condition) and after (body)
          ; before region ends with scf.condition
          ; after region ends with scf.yield
          (def result (s/while {:result i64} init
            ; before region - must end with scf.condition
            (region
              (block [(: arg i64)]
                (def cond (a/cmpi {:predicate "slt"} arg bound))
                (s/condition cond arg)))
            ; after region - must end with scf.yield
            (region
              (block [(: arg i64)]
                (def next (a/addi arg c1))
                (s/yield next)))))
          (f/return result))))

    ; Test 2: scf.while with multiple iter args
    ; NOTE: This test exposes a gap - multi-result destructuring (def (a b) ...) not supported
    ; Would need syntax like: (def result (s/while ...)) and then some way to extract results
    (f/func {:sym_name "fib_while"
             :function_type (-> [i64] [i64])
             :llvm.emit_c_interface true}
      (region
        (block [(: n i64)]
          (def c0 (: 0 i64))
          (def c1 (: 1 i64))
          ; Simplified: just count up
          (def result (s/while {:result i64} c0
            (region
              (block [(: i i64)]
                (def cond (a/cmpi {:predicate "slt"} i n))
                (s/condition cond i)))
            (region
              (block [(: i i64)]
                (def next_i (a/addi i c1))
                (s/yield next_i)))))
          (f/return result))))))
