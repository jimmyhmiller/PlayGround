;; Comprehensive demo using all macros: defn, constant, +, *, return
;; Demonstrates a single function using all arithmetic macros

;; Function: compute() = (10 + 5) * 2 = 30
(defn main [] i64
  (return
    (* (: i64)
      (+ (: i64)
        (constant (: 10 i64))
        (constant (: 5 i64)))
      (constant (: 2 i64)))))
