(ns testing-stuff.diamond 
  (:require [clojure.string :as string]))

(defn range-with-pivot [pivot start step]
  (let [common-seq (range start pivot step)]
    (concat common-seq [pivot] (reverse common-seq))))

(defn space-count [count]
  (range-with-pivot 0 (dec count) -1))

(defn letter-count [count]
  (range-with-pivot (dec (* count 2)) 1 2))

(defn letter-seq [c]
  (map char (range-with-pivot (int c) (int \A) 1)))

(defn repeat-str [n str]
  (string/join (repeat n str)))

(defn letter->int [char]
  (- (int char) 64))

(defn diamond [letter]
  (let [char (first letter)
        num (letter->int char)
        spaces (map repeat-str (space-count num) (repeat " "))
        letters (map repeat-str (letter-count num) (letter-seq char))]
    (->> (map str spaces letters)
         (string/join "\n"))))
