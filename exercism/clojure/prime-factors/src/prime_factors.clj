(ns prime-factors)

(defn factors2 [n]
  (cond
    (= n 1) []
    (zero? (mod n 2)) (conj (factors2 (/ n 2)) 2)
    :else [n]))

(defn factors-starting-at [f n]
  (cond
    (= n 1) []
    (> f (Math/sqrt n)) [n]
    (zero? (mod n f)) (cons f (factors-starting-at f (/ n f)))
    :else (recur (+ f 2) n)))

(defn of [n]
  (if (= 1 n)
    '()
    (let [facts (factors2 n)
          n (first facts)]
      (concat (rest facts) (factors-starting-at 3 n)))))

