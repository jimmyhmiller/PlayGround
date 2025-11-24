; Memory operations example using memref dialect

(require-dialect [memref :as m] [arith :as a] [func :as f] [scf :as s])

(module
  (do
    ; Initialize an array with zeros
    (f/func {:sym_name "init_array"
             :function_type (-> [memref<10xi64>] [])}
      (do
        (block [(: buffer memref<10xi64>)]
          (def zero (: 0 i64))
          (def c0 (: 0 index))
          (def c10 (: 10 index))
          (def c1 (: 1 index))

          (s/for {:lower_bound c0
                  :upper_bound c10
                  :step c1}
            (do
              (block [(: i index)]
                (m/store zero buffer i))))

          (f/return))))

    ; Sum array elements
    (f/func {:sym_name "sum_array"
             :function_type (-> [memref<10xi64>] [i64])}
      (do
        (block [(: buffer memref<10xi64>)]
          (def zero (: 0 i64))
          (def c0 (: 0 index))
          (def c10 (: 10 index))
          (def c1 (: 1 index))

          (def sum
            (s/for {:lower_bound c0
                    :upper_bound c10
                    :step c1}
              zero
              (do
                (block [(: i index) (: acc i64)]
                  (def val (m/load buffer i))
                  (def new_acc (a/addi acc val))
                  (s/yield new_acc)))))

          (f/return sum))))

    ; Main function
    (f/func {:sym_name "main"
             :function_type (-> [] [i64])}
      (do
        (block []
          (def buffer (m/alloc))
          (f/call "init_array" buffer)
          (def result (f/call "sum_array" buffer))
          (m/dealloc buffer)
          (f/return result))))))
