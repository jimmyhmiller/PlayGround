(ns prime-multiplication.core
  (:require [clojure.string :as string]
            [clojure.pprint :refer [cl-format]]))

(defn prime-divisors? [n primes]
  (boolean (some #(zero? (mod n %)) primes)))

(defn prime? [n previous-primes]
  (cond 
    (< n 2) false
    (= n 2) true
    (zero? (mod n 2)) false
    :else (not (prime-divisors? n previous-primes))))

(def primes
  "Loosely based off:
   https://github.com/stuarthalloway/programming-clojure/blob/master/src/examples/primes.clj

   The wheel is much more clever than my solution, I mainly looked
   at this version to make sure I was getting my lazy-seq right.
   After getting the basic structure though, I wrote the rest
   from scratch (hence how inelegent it is compared to that version)"
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

(defn calculate-spacing
  "Calculates spacing for all elements given n. This may not
   be obvious how it works, but if you think about it,
   the largest number we will have is n^2. This finds
   the length of that number and makes sure it will
   have at least one space next to it. Without this, numbers
   might clump together."
  [n]
  (+ 1 (count (str (* n n)))))

(defn format-row [formatter row]
  (->> row
       (map formatter)
       string/join))

(defn parse-int [n]
  (Integer/parseInt n))

(defn make-table [n]
  (let [all-numbers (cons 1 (take n primes))
        spacing (calculate-spacing (last all-numbers))
        formatter (memoize (partial format-element spacing))]
    (->> all-numbers
         multiples
         (partition (inc n))
         (pmap (partial format-row formatter))
         (string/join "\n"))))

(defn -main
  [& [upper-bound]]
  (println (make-table (parse-int (or upper-bound "10"))))
  (shutdown-agents))
