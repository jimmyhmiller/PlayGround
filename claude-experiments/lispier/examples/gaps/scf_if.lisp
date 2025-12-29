; WORKING: scf.if operation
; scf.if is different from cf.cond_br - it's a structured if-then-else with results
; Status: scf.if works correctly (both with and without else regions)

(require-dialect [func :as f] [arith :as a] [scf :as s] [memref :as m])

(module
  (do
    ; Test 1: Basic scf.if with result
    ; In MLIR:
    ; %result = scf.if %cond -> (i32) {
    ;   scf.yield %true_val : i32
    ; } else {
    ;   scf.yield %false_val : i32
    ; }
    (f/func {:sym_name "scf_if_basic"
             :function_type (-> [i1] [i32])
             :llvm.emit_c_interface true}
      (region
        (block [(: cond i1)]
          (def true_val (: 42 i32))
          (def false_val (: 0 i32))
          ; scf.if needs two regions: then and else
          ; and it produces a result
          (def result (s/if {:result i32} cond
            (region
              (block []
                (s/yield true_val)))
            (region
              (block []
                (s/yield false_val)))))
          (f/return result))))

    ; Test 2: scf.if without else (no result)
    ; This now works correctly!
    (f/func {:sym_name "scf_if_no_else"
             :function_type (-> [i1 memref<1xi32>] [])}
      (region
        (block [(: cond i1) (: mem memref<1xi32>)]
          (def val (: 99 i32))
          (def idx (: 0 index))
          ; scf.if with only then region, no result
          (s/if cond
            (region
              (block []
                (m/store val mem idx)
                (s/yield))))
          (f/return))))))