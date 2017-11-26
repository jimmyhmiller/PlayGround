(ns account-number.core
  (:require [clojure.string :as string]
            [cuerdas.core :refer [<<-]]))
(def reference-digits
  (<<- " _     _  _     _  _  _  _  _ 
        | |  | _| _||_||_ |_   ||_||_|
        |_|  ||_  _|  | _||_|  ||_| _|"))

(defn split-each-row
  "split-each-row is a helper function
   that takes a collection of strings
   and splits them per character. This
   allows us to break up the digits into
   spaces, pipes, and underscores."
  [row]
  (->> row
       (map #(string/split % #""))
       (into [])))

(defn split-into-rows
  "split-into-rows converts a string
   into a list of 2d vectors allowing
   random access in order to pull out
   the digits"
  [contents]
  (->> (string/split contents #"\n")
       (partition 3)
       (map split-each-row)))

(defn get-ascii-digit 
  "Takes a 2d vector and an int 
   returns a flattened 3x3 collection.
   Allows you to pull out all the segments
   of a particular digit."
  [rows x]
  (for [y (range 0 3)
        x (range (* x 3) (+ (* x 3) 3))]
    (get-in rows [y x])))

(defn remove-unnecessary-segments
  "The ascii art of this program is
   converted to seven segment displays.
   The first and third characters of any
   digit will always be spaces. So we
   remove them giving us only the seven
   segments."
  [[_ a _ & rest]]
  (into [a] rest))

(defn to-seven-segment 
  "A seven segment display is represented 
   as a vector of booleans.
   Each segment in a seven segment display is
   either on or off. In this case anything that
   isn't a space is considered on (true)."
  [ascii-digit]
  (->> ascii-digit
       remove-unnecessary-segments
       (map #(not= " " %))
       (into [])))

(defn rows->seven-segment
  "Converts each row of ascii digits
   into seven segment displays."
  [rows]
  (->> (range (/ (count (first rows)) 3))
       (map (partial get-ascii-digit rows))
       (map to-seven-segment)))

(def valid-seven-segments
  "A list of seven segment displays for
   all the valid digits."
  (->> reference-digits
       split-into-rows
       (map rows->seven-segment)
       first
       (into [])))

(defn int->seven-segment
  "Using valid-seven-segments this function
   allows you to convert from an int to
   a seven-segment display."
  [n]
  (get valid-seven-segments n))

(def seven-segment-to-int-map
  "This is the opposite mapping compared to
   valid-seven-segments."
  (->> (range)
       (map vector valid-seven-segments)
       (into {})))

(defn seven-segment->int
  "Given a seven segment display you can
   get the corresponding integer."
  [segment]
  (get seven-segment-to-int-map segment))

(defn seven-segment->account-number [coll]
  (mapv seven-segment->int coll))

(defn check-sum [numbers]
  (let [sum (->> (range 1 10)
                 (map * (reverse numbers))
                 (reduce +))]
    (mod sum 11)))

(defn has-invalid-digits? [numbers]
  (not-every? (complement nil?) numbers))

(defn valid-account-number? [numbers]
  (and (not (has-invalid-digits? numbers))
       (zero? (check-sum numbers))))

(defn determine-error-message [numbers]
  (let [has-invalid-numbers (has-invalid-digits? numbers)
        is-valid (valid-account-number? numbers)]
    (cond
      has-invalid-numbers " ILL" 
      (not is-valid) " ERR" 
      :else "")))

(defn format-output [numbers]
  (let [error-message (determine-error-message numbers)]
    (as-> numbers n
      (map (fnil identity "?") n)
      (string/join n)
      (str n error-message))))

(defn hamming-distance [segment1 segment2]
  {:pre [(= (count segment1) (count segment2))]}
  (->> (map vector segment1 segment2)
       (filter (fn [[x y]] (not= x y)))
       count))

(defn find-possible-digits [index segment]
  (->> valid-seven-segments
       (filter #(= (hamming-distance segment %) 1))
       (map seven-segment->int)
       (map vector (repeat index))))

(defn find-replacements [account-number]
  (->> (map-indexed find-possible-digits account-number)
       (mapcat identity)))

(defn try-possibility [account-number [index value]]
  (assoc account-number index value))

(defn generate-possible-account-numbers [account-number-segments]
  (let [account-number (seven-segment->account-number account-number-segments)
        replacements (find-replacements account-number-segments)]
    (->> (map (partial try-possibility account-number) replacements)
         (filter valid-account-number?))))
