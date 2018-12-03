(ns advent.core
  (:require [clojure.string :as string]
            [clojure.set :as set]))


;;; Challenge 1

(defn challenge1-part1 []
  (->> (slurp "challenges/1.txt")
       (string/split-lines)
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
  (->> (slurp "challenges/1.txt")
       (string/split-lines)
       (map read-string)
       (cycle)
       (reduce first-duplicate [#{0} 0])))


;;; Challenge 2

(defn challenge2-part1 []
  (->> (slurp "challenges/2.txt")
       (string/split-lines)
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
  (->> (slurp "challenges/2.txt")
       (string/split-lines)
       (find-correct-ids)
       (filter-non-match-char)))



;; Challenge 3

(defn square [{:keys [x y width height]}]
  (for [i (range x (+ x width))
        j (range y (+ y height))]
    [i j]))

(defn parse-coords [coords]
  (->> coords
       string/split-lines
       (map #(string/split % #":? |,|x"))
       (map (fn [[n _ x y w h]] 
              {:n (subs n 1) 
               :x (read-string x) 
               :y (read-string y)
               :width (read-string w) 
               :height (read-string h)}))))

(defn challenge3-part1 []
  (->> (slurp "challenges/3.txt") 
       parse-coords
       (map (juxt :n square))
       (mapcat square)
       frequencies
       vals
       (filter #(> % 1))
       count))

(defn in-points [points square]
  (every? (partial contains? points) square))

(defn challenge3-part2 []
  (let [squares (->> (slurp "challenges/3.txt") 
                     parse-coords
                     (map (juxt :n square)))
        non-overlap (->> squares
                         (mapcat second)
                         frequencies 
                         (filter (comp #{1} second))
                         (map first)
                         set)]
    (->> squares
         (filter (comp (partial in-points non-overlap) second))
         ffirst)))


(comment
  (time
   [(challenge1-part1)
    (challenge1-part2)
    (challenge2-part1)
    (challenge2-part2)
    (challenge3-part1)
    (challenge3-part2)]))
