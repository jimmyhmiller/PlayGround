(ns luhn
  (:require [clojure.string :refer [replace join]]))

(defn char->int [char]
  (read-string (str char)))

(defn int->char [int]
  (-> int
      str
      char-array
      first))

(defn double-digit [n]
  (let [doubled (* (char->int n) 2)]
    (int->char
     (if (> doubled 9)
       (- doubled 9)
       doubled))))

(defn all-nums [num]
  (every? #(Character/isDigit %) num))

(defn reverse-num [num]
  (join "" (reverse num)))

(defn clean-num [num]
  (replace (reverse-num num) #" " ""))

(defn sum-digits [cleaned-num]
  (let [even-digits (take-nth 2 cleaned-num)
        odd-digits (take-nth 2 (drop 1 cleaned-num))
        doubled-digits (map double-digit odd-digits)]
    (->> (mapcat list even-digits doubled-digits)
         (map char->int)
         (reduce +))))

(defn divisible? [n div]
  (zero? (mod n div)))

(defn valid? [num]
  (let [cleaned-num (clean-num num)]
    (cond
      (= (count cleaned-num) 1) false
      (not (all-nums cleaned-num)) false
      :else (divisible? (sum-digits cleaned-num) 10))))


