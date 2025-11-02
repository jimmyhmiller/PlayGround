;; Demo of defn macro
;; Defines a function that computes (a + b) * 2

(defn main [(: %a i64) (: %b i64)] i64
  (return
    (* (: i64)
      (+ (: i64) %a %b)
      (constant (: 2 i64)))))
