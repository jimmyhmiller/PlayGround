(def f (: (-> [Int] Int))
  (fn [x] (+ x 1)))

(def result (: Int) (f 1 2))
