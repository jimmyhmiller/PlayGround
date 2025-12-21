; Structured control flow example using SCF dialect
; Simplified version with proper scoping

(require-dialect [scf :as s] [arith :as a] [func :as f])

(module
  (do
    ; Simple loop that counts from 0 to 9 and returns final count
    (f/func {:sym_name "count_to_10"
             :function_type (-> [] [index])
             :llvm.emit_c_interface true}
      (region
        (block []
          (def c0 (: 0 index))
          (def c10 (: 10 index))
          (def c1 (: 1 index))

          ; scf.for returns a value
          (def result (s/for {:result index} c0 c10 c1 c0
            (region
              (block [(: i index) (: acc index)]
                (def new_acc (a/addi acc c1))
                (s/yield new_acc)))))

          (f/return result))))

    ; Simple conditional
    (f/func {:sym_name "sign"
             :function_type (-> [i64] [i64])
             :llvm.emit_c_interface true}
      (region
        (block [(: x i64)]
          (def zero (: 0 i64))
          (def one (: 1 i64))
          (def neg_one (: -1 i64))
          (def is_pos (a/cmpi {:predicate "sgt"} x zero))

          ; Use simple operations without complex scf
          (def result (arith.select is_pos one neg_one))
          (f/return result))))))
