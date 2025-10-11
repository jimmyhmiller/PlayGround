(def factorial (: (-> [Int] Int))
  (fn [n]
    (if (= n 0)
      1
      (* n (factorial (- n 1))))))

(def result (: Int) (factorial 5))
(printf (c-str "%lld\n") result)
