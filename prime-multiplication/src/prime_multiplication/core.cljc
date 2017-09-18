(ns prime-multiplication.core
  (:require [clojure.string :as string]
            [clojure.pprint :refer [cl-format]]
            [clojure.core.reducers :as r]))

(defn prime-divisors? [n primes]
  (boolean (some #(zero? (mod n %)) primes)))

(defn prime? [n previous-primes]
  (cond 
    (< n 2) false
    (= n 2) true
    (zero? (mod n 2)) false
    :else (not (prime-divisors? n previous-primes))))

(def primes
  (concat 
   [2 3 5 7]
   (lazy-seq
    (let [next-prime
          (fn next-prime [n]
            (if (prime? n (take-while #(<= n (* % %)) primes))
              (lazy-seq (cons n (next-prime (+ n 2))))
              (recur (+ n 2))))]
      (next-prime 11)))))

(defn multiples [coll]
  (for [p1 coll p2 coll]
    (* p1 p2)))

(defn format-element [n val]
  (if (= val 1) 
    (string/join (repeat n " "))
    (cl-format nil (str "~" n "' d") val)))

(defn calculate-spacing [n]
  (+ 1 (count (str (* n n)))))

(defn format-row [formatter row]
  (->> row
       (map formatter)
       string/join))

(defn parse-int [n]
  #?(:clj (Integer/parseInt n)
     :cljs (js/parseInt n)))

(defn -main
  [& [upper-bound]]
  (let [n (parse-int (or upper-bound "10"))
        all-numbers (cons 1 (take n primes))
        spacing (calculate-spacing (last all-numbers))
        mapper #?(:clj pmap :cljs map)
        formatter (memoize (partial format-element spacing))]
    (->> all-numbers
         multiples
         (partition (inc n))
         (mapper (partial format-row formatter))
         (string/join "\n")
         println))
  #?(:clj (shutdown-agents)))


