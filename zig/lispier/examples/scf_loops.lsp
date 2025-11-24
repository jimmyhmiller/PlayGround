; Structured control flow example using SCF dialect

(require-dialect [scf :as s] [arith :as a] [func :as f])

(module
  (do
    ; Simple if-else
    (f/func {:sym_name "abs"
             :function_type (-> [i64] [i64])}
      (do
        (block [(: x i64)]
          (def zero (: 0 i64))
          (def is_neg (a/cmpi {:predicate "slt"} x zero))

          (def result
            (s/if is_neg
              (do
                (def neg (a/subi zero x))
                (s/yield neg))
              (do
                (s/yield x))))

          (f/return result))))

    ; For loop with accumulator
    (f/func {:sym_name "sum_range"
             :function_type (-> [i64 i64] [i64])}
      (do
        (block [(: start i64) (: end i64)]
          (def zero (: 0 i64))
          (def one (: 1 i64))

          (def sum
            (s/for {:lower_bound start
                    :upper_bound end
                    :step one}
              zero
              (do
                (block [(: i i64) (: acc i64)]
                  (def new_acc (a/addi acc i))
                  (s/yield new_acc)))))

          (f/return sum))))

    ; While loop
    (f/func {:sym_name "factorial"
             :function_type (-> [i64] [i64])}
      (do
        (block [(: n i64)]
          (def one (: 1 i64))
          (def zero (: 0 i64))

          (let [[counter result]
                (s/while
                  (do
                    (block [(: c i64) (: r i64)]
                      (def cond (a/cmpi {:predicate "sgt"} c zero))
                      (s/condition cond c r)))
                  (do
                    (block [(: c i64) (: r i64)]
                      (def new_r (a/muli r c))
                      (def new_c (a/subi c one))
                      (s/yield new_c new_r)))
                  n one)]
            (f/return result)))))))
