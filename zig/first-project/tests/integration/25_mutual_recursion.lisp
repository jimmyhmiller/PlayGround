(def is_even (: (-> [Int] Bool))
  (fn [n]
    (if (= n 0)
      true
      (is_odd (- n 1)))))

(def is_odd (: (-> [Int] Bool))
  (fn [n]
    (if (= n 0)
      false
      (is_even (- n 1)))))

(def result1 (: Int) (if (is_even 4) 1 0))
(def result2 (: Int) (if (is_odd 5) 1 0))
(printf (c-str "%d %d\n") result1 result2)
