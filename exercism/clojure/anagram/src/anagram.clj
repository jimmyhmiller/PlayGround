(ns anagram
  (:require [clojure.string :refer [lower-case]]))

(defn anagram? [word1 word2]
  (= (frequencies word1)
     (frequencies word2)))

(defn proper-anagram? [word1 word2]
  (and (not= word1 word2)
       (anagram? word1 word2)))

(defn anagrams-for [word words]
  (let [lower-word (lower-case word)]
    (filter (comp (partial proper-anagram? lower-word) lower-case) words)))

