(def add3 (: (-> [Int Int Int] Int))
  (fn [a b c]
    (+ a (+ b c))))

(def result (: Int) (add3 10 20 30))
(printf (c-str "%lld\n") result)
