(defn true' [t f] t)
(defn false' [t f] f)
(defn if' [p t f] (p t f))
(defn and' [p1 p2] (if' p1 (if' p2 true' false') false'))
(defn cond' [p1 t1 p2 t2 p3 t3 else t4]
  (if' p1 t1
    (if' p2 t2
      (if' p3 t3
           t4))))
(def mods [true' false' false' false' false'])
(defn divisible? [n m] (get mods (mod n m)))


(defn fizzbuzz []
  (for [i (range 1 101)]
    (cond'
      (and' (divisible? i 3) (divisible? i 5)) "fizzbuzz"
      (divisible? i 3) "fizz"
      (divisible? i 5) "buzz"
      :else i)))

