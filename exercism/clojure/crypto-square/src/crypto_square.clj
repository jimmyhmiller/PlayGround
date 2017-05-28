(ns crypto-square
  (:require [clojure.string :refer [replace join lower-case]]))

(defn square-size [text]
  (int (Math/ceil (Math/sqrt (count text)))))

(defn normalize-plaintext [text]
  (replace (lower-case text) #"[\p{Punct} ]" ""))

(defn breakup-text [text]
  (let [cleaned-text (normalize-plaintext text)
        split-size (square-size cleaned-text)]
    (partition split-size split-size (repeat "") cleaned-text)))

(defn plaintext-segments [text]
  (->> text
       breakup-text
       (map #(join "" %))))

(defn ciphertext [text]
  (->> text
       breakup-text
       (apply map list)
       flatten
       (join "")))
