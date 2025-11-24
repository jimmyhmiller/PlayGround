; Control flow example with branches and loops

(require-dialect [cf :as c] [arith :as a] [func :as f])

(module
  (do
    ; Max function using conditional branch
    (f/func {:sym_name "max"
             :function_type (-> [i64 i64] [i64])}
      (do
        (block [(: x i64) (: y i64)]
          (def cond (a/cmpi {:predicate "sgt"} x y))
          (c/cond_br {:successors [^return_x ^return_y]
                      :operand_segment_sizes [1 0 0]}
                     cond))

        (block ^return_x
          (f/return x))

        (block ^return_y
          (f/return y))))

    ; Countdown loop using block arguments
    (f/func {:sym_name "countdown"
             :function_type (-> [i64] [i64])}
      (do
        (block [(: n i64)]
          (c/br {:successors [^loop]} n))

        (block ^loop [(: iter i64)]
          (def is_zero (a/cmpi {:predicate "eq"} iter 0))
          (c/cond_br {:successors [^done ^continue]
                      :operand_segment_sizes [1 0 1]}
                     is_zero iter))

        (block ^continue [(: val i64)]
          (def next (a/subi val 1))
          (c/br {:successors [^loop]} next))

        (block ^done
          (f/return 0))))

    ; Sum from 1 to n
    (f/func {:sym_name "sum_to_n"
             :function_type (-> [i64] [i64])}
      (do
        (block [(: n i64)]
          (def zero (: 0 i64))
          (c/br {:successors [^loop]} n zero))

        (block ^loop [(: counter i64) (: acc i64)]
          (def is_zero (a/cmpi {:predicate "eq"} counter 0))
          (c/cond_br {:successors [^done ^continue]
                      :operand_segment_sizes [1 1 2]}
                     is_zero acc counter acc))

        (block ^continue [(: c i64) (: sum i64)]
          (def new_sum (a/addi sum c))
          (def next_c (a/subi c 1))
          (c/br {:successors [^loop]} next_c new_sum))

        (block ^done [(: final i64)]
          (f/return final))))))
