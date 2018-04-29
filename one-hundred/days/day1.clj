(ns day1
  (:require [clojure.string :as string]))


(defn logic-variable? [sym]
  (and (symbol? sym)
       (string/starts-with? (name sym) "?")))

(defn lookup [var var-map]
  (if-let [val (var-map var)]
    (if (logic-variable? val)
      (recur val var-map)
      val)
    var))

(defn add-var [var val var-map]
  (when var-map
    (assoc var-map var val)))

(defn unify [x y var-map]
  (if (map? var-map)
      (let [x' (lookup x var-map)
            y' (lookup y var-map)]
        (cond 
          (= x' y') var-map
          (logic-variable? x') (add-var x' y' var-map)
          (logic-variable? y') (add-var y' x' var-map)
          :else :unify/failed))
      var-map))

(defn substitute [vars var-map]
  (map #(lookup % var-map) vars))


(substitute '[?x ?y ?z]
            (->> {}
                 (unify '?x 1)
                 (unify '?y 2)
                 (unify '?z 3)))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;




(def TRUE (fn [t f] t))
(def FALSE (fn [t f] f))
(def IF (fn [pred t f] (pred t f)))

(def AND 
  (fn [a b]
    (IF a (IF b TRUE FALSE) FALSE)))

(defn is-zero? [n]
  (nth (cons TRUE (repeat FALSE)) n))

(defn evenly-divisible? [n m]
  (is-zero? (mod n m)))

(defn COND [p1 t1 p2 t2 p3 t3 _ e]
  (IF p1 t1 (IF p2 t2 (IF p3 t3 e))))

(defn fizz [n]
  (COND 
   (AND (evenly-divisible? n 3)
        (evenly-divisible? n 5)) "FizzBuzz"
   (evenly-divisible? n 3)  "Fizz"
   (evenly-divisible? n 5) "Buzz" 
   :else n))


(map fizz (range 1 101))





