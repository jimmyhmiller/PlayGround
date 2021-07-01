(ns query.core)

;; https://scattered-thoughts.net/writing/a-practical-relational-query-compiler-in-500-lines/



(defn gallop [coll val]
  (if (and (not (empty? coll))
           (< (first coll) val))
    (let [step 1
          result
          (loop [coll coll val val step step]
            (if (and (< step (count coll))
                     (< (nth coll step) val))
              (recur (subvec coll step)
                     val
                     (bit-shift-left step 1))
              ;; Would love to get rid of this allocation
              ;; I could consider deftype with a mutable variable for this stuff
              [coll step]))
          coll (nth result 0)
          step (nth result 1)
          step (bit-shift-right step 1)
          coll (loop [coll coll
                      val val
                      step step]
                 (if (> step 0)
                   (if (and (< step (count coll))
                            (< (nth coll step) val))
                     (recur (subvec coll step)
                            val
                            (bit-shift-right step 1))
                     (recur coll val (bit-shift-right step 1)))
                   coll))]
      (subvec coll 1)))
  coll)



(gallop [9 10] 1)


(defn intersect [column-a column-b]
  (loop [column-a column-a
         column-b column-b
         results []]
    (if (empty? column-a)
      results
      (let [value-a (first column-a)
            column-b (gallop column-b value-a)
            value-b (first column-b)]
        (if (and value-a (= value-a value-b))
          ;; So in the example, they gallop with a > predicate
          ;; But why? Maybe they allow duplicates?
          ;; I mean that would make sense in that case.
          ;; If I allow duplicates I should probably do that.
          (recur (subvec column-a 1)
                 (subvec column-b 1)
                 (conj results value-a))
          (recur column-b
                 column-a
                 results))))))








(- (count [3 4 5]) (count [2 3]) )
(ffirst [[1 2 3] [3 4 5]])

(intersect* [[1 2 3] [3 4 5]]
            [[1 2 3] [3 4 5]])



(subvec [3 4 5] 2)




;; Okay, so I think I got some stuff a bit wrong.
;; It seems that indexes might be tuples of arrays?
;; So that means I need to gallop to find the values
;; and I would really need to deal with indexes instead of subvecs?


[0 :playlist/name "Heavy Metal Classic"]
[1 :playlist-track/playlist 0]
[1 :playlist-track/track 2]
[2 :track/album 3]
[3 :album/artist 4]
[4 :artist/name "Death"]


;; This is mixing up different kinds of indexes
;; We can deal with that fact later.

(def indexes
  {:playlist/name {"Heavy Metal Classic" [0]}
   :playlist-track/playlist [[0] [1]]
   :playlist-track/track {1 [2]}
   :track/album {2 [3]}
   :album/artist {3 [4]}
   :artist/name {4 ["Death"]}})



(intersect (get indexes))




(let [upper-1 (rand-int 100000)
      lower-1 (rand-int upper-1)
      upper-2 (+ lower-1 (rand-int upper-1))
      lower-2 (rand-int upper-2)
      upper-3 (+ lower-2 (rand-int upper-2))
      lower-3 (rand-int upper-3)
      column-a (into [] (range lower-1 upper-1))
      column-b (into [] (range lower-2 upper-2))
      column-c (into [] (range lower-3 upper-3))]
  (println (+ (count column-a) (count column-b) (count column-c)))
  (println lower-1 upper-1 lower-2 upper-2 lower-3 upper-3)
  (time
   (do
     (println
      (count
       (intersect column-c
                  (intersect column-a column-b))))
     nil)))
