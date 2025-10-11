(def apply_twice (: (-> [(-> [Int] Int) Int] Int))
  (fn [f x]
    (f (f x))))

(def add_one (: (-> [Int] Int))
  (fn [n] (+ n 1)))

(def result (: Int) (apply_twice add_one 10))
(printf (c-str "%lld\n") result)
