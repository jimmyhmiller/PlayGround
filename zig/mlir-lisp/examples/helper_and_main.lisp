;; Demo showing multiple function definitions with a helper function
;; Helper function adds two numbers
;; Main function computes (10 + 5) * 2 = 30 using the helper

(defn add_numbers [(: %a i64) (: %b i64)] i64
  (return
    (+ (: i64) %a %b)))

(defn main [] i64
  (return
    (* (: i64)
      (call @add_numbers (constant (: 10 i64)) (constant (: 5 i64)) i64)
      (constant (: 2 i64)))))
