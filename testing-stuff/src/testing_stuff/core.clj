(ns testing-stuff.core
  (:use [clojure.math.numeric-tower :only [sqrt]]))



(defn list-of-primes
  "List of primes up to n.
  Implemented for speed not beauty."
  [n]
  (let [sieve (boolean-array n true)]
    (doseq [p (range 3 (int (sqrt n)) 2)]
      (when (aget sieve p)
        (doseq [i (range (* p p) n (* p 2))]
          (aset sieve i false))))
    (cons 2 (filter #(aget sieve %)(range 3 n 2)))))


(defn cl []
  (fn [m & args]
    (let [public
          {:x (fn [n v] (* n v))}]
      (apply (m public) args))))


(def x (cl))

(x :x 2 3)

(defn square [x] (* x x))

(time (list-of-primes 100000))

(defn divisible? [a b]
  (zero? (mod a b)))


(def primes
  (cons 2
        (lazy-seq (filter prime? (iterate inc 3)))))


(defn prime? [n]
  (loop [ps primes]
    (cond (> (square (first ps)) n) true
          (divisible? n (first ps)) false
          :else (recur (rest ps)))))


(time (last (take 10000 primes)))


(defn sieve [s]
  (cons (first s)
        (lazy-seq (sieve
                   (filter #(not (divisible? % (first s)))
                   (rest s))))))

(time (last (take 10000 (sieve (iterate inc 2)))))
