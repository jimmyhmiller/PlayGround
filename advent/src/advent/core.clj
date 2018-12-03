(ns advent.core
  (:require [clojure.string :as string]
            [clojure.set :as set]))


;;; Challenge 1

(defn challenge1-part1 []
  (->> (string/split (slurp "challenges/1.txt") #"\n")
       (map read-string)
       (reduce +)))


;; I have since learned I could have used reductions.
;; Didn't know that existed
;; Need to reformulate with that.
(defn first-duplicate [[history total] val]
  (let [answer (+ total val)]
    (if (contains? history answer)
      (reduced answer)
      [(conj history answer) answer])))


(defn challenge1-part2 []
  (->> (string/split (slurp "challenges/1.txt") #"\n")
       (map read-string)
       (cycle)
       (reduce first-duplicate [#{0} 0])))


;;; Challenge 2

(defn challenge2-part1 []
  (->> (string/split (slurp "challenges/2.txt") #"\n")
       (map frequencies)
       (map vals)
       (map set)
       (mapcat (partial filter #{2 3}))
       (frequencies)
       (vals)
       (apply *)))


(defn hamming-distance [coll1 coll2]
  {:pre [(= (count coll1) (count coll2))]}
  (->> (map vector coll1 coll2)
       (filter (fn [[x y]] (not= x y)))
       count))

(defn find-correct-ids [ids]
  (->> (for [id1 ids
             id2 ids]
         [id1 id2 (hamming-distance id1 id2)])
       (filter (comp #{1} last))
       (first)
       (take 2)))

(defn filter-non-match-char [[word1 word2]]
  (->> (map vector word1 word2)
       (filter (partial apply =))
       (map first)
       (string/join)))

(defn challenge2-part2 []
  (let [ids (string/split (slurp "challenges/2.txt") #"\n")]
    (->> (find-correct-ids ids)
         (filter-non-match-char))))


(comment
  (time
   [(challenge1-part1)
    (challenge1-part2)
    (challenge2-part1)
    (challenge2-part2)]))
