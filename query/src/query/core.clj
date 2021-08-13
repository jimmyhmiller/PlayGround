(ns query.core)

;; https://scattered-thoughts.net/writing/a-practical-relational-query-compiler-in-500-lines/



(defn gallop [coll low high compare val ]
  (let [compare (complement compare)]
    (if (and (< low high)
             (compare (nth coll low) val))
      (let [step 1
            result (loop [i low val val step step]
                     (if (and (< (+ i step) high)
                              (compare (nth coll (+ i step)) val))
                       (recur (+ i step)
                              val
                              (bit-shift-left step 1))
                       [i step]))
            i (nth result 0)
            step (nth result 1)
            step (bit-shift-right step 1)
            i (loop [i i
                     val val
                     step step]
                
                (if (> step 0)
                  (if (and (< (+ i step) high)
                           (compare (nth coll (+ i step)) val))
                    (recur (+ i step)
                           val
                           (bit-shift-right step 1))
                    (recur i val (bit-shift-right step 1)))
                  i))]
        (inc i))
     low)))

;; The other example is lazy, should this be lazy?

;; I think I also need this to work on tuples. But I can just change > and >= to work with some tuples.
;; Then I should get back the tuple with the right value. I think that is how this works, still a bit confused.
;; Actualy, they are tuples of columns
;; So how does all of this work?
;; Maybe instead of returning the value-a I found I need to be returning the value-a of the other index?
;; I just really don't understand how to put these pieces together.
;; Or maybe this does make sense?
;; See below for an example that is seeming right
(defn intersect [column-a column-b]
  (loop [column-a column-a
         column-b column-b
         index-a 0
         index-b 0
         result []
         gas 0]
    
    (if (or (>= index-a (count column-a))
           #_ (>= gas 100))
      result
      (let [value-a (nth column-a index-a)
            index-b (gallop column-b index-b (count column-b) >= value-a)
            value-b (if (>= index-b (count column-b) )
                      ::not-matching
                      (nth column-b index-b))]
        (if (= value-a value-b)
          (recur column-a
                 column-b
                 (gallop column-a index-a (count column-a) > value-a)
                 (gallop column-b index-b (count column-b) > value-b)
                 (conj result value-a)
                 (inc gas))
          (recur column-b
                 column-a
                 index-b
                 index-a
                 result
                 (inc gas)))))))







(def query '[[?playlist :playlist/name "Heavy Metal Classic"]
             [?playlist :playlist/track ?track]
             [?track :track/name ?track-name]
             [?track :track/album ?album]
             [?album :album/name ?album-name]
             [?album :album/artist ?artist]
             [?artist :artist/name ?artist-name]])


[0 :playlist/name "Heavy Metal Classic"]
[1 :playlist-track/playlist 0]
[1 :playlist-track/track 2]
[2 :track/album 3]
[3 :album/artist 4]
[4 :artist/name "Death"]
[]


(def indexes 
  {:eav {0 {:playlist/name ["Heavy Metal Classic"]
            :playlist/track [1 2 3]}
         1 {:track/name ["Track 1"]
            :track/album [4]}
         2 {:track/name ["Track 2"]
            :track/album [4]}
         3 {:track/name ["Track 3"]
            :track/album [5]}
         4 {:album/name ["Debut"]
            :album/artist [6]}
         5 {:album/name ["Debut 2"]
            :album/artist [6]}
         6 {:artist/name ["Great Band Name"]}}
   :aev {:playlist/name {0 ["Heavy Metal Classic"]}
         :track/name {1 ["Track 1"]
                      2 ["Track 2"]
                      3 ["Track 3"]}
         :track/album {1 [4]
                       2 [4]
                       3 [5]}
         :album/name {4 ["Debut"]
                      5 ["Debut 2"]}
         :album/artist {4 [6]
                        5 [6]}
         :artist/name {6 ["Great Band Name"]}}
   :vae {"Heavy Metal Classic" {:playlist/name [0]}
         "Track 1" {:track/name [1]}
         "Track 2" {:track/name [2]}
         "Track 3" {:track/name [3]}
         4 {:track/album [1 2]}  
         5 {:track/album [3]}
         "Debut" {:album/name [4]}
         "Debut 2" {:album/name [5]}
         6 {:album/artist [4 5]}
         "Great Band Name" {:artist/name [6]}}})


(defn logic-var? [x]
  (and (symbol? x)
       (clojure.string/starts-with? (name x) "?")))


(defn constant? [x]
  (not (logic-var? x)))

(defn query-frequencies [query]
  (->> query
       (mapcat identity)
       (filter logic-var?)
       frequencies))

;; Very wrong function for bound, but going with it till I know something better
(defn l-bound? [query x]
  (let [freq (query-frequencies query)]
    (or (not (logic-var? x))
        (> (get freq x 0) 1))))



;; Needs to be better

(defn triple-to-index [query [e a v]]
  (cond
    (and (constant? v) (constant? a))
    [e :vae v a]
    (and (constant? e) (constant? a))
    [v :eav e a]
    (and (l-bound? query e) (l-bound? query a))
    [v :eav e a]
    (and (l-bound? query v) (l-bound? query a))
    [e :vae v a]
    :else (throw (ex-info "not covered" {:index [e a v]}))))


(map (partial triple-to-index query) query)

;; =>

([?playlist :vae "Heavy Metal Classic" :playlist/name]
 [?track :eav ?playlist :playlist/track]
 [?track-name :eav ?track :track/name]
 [?album :eav ?track :track/album]
 [?album-name :eav ?album :album/name]
 [?artist :eav ?album :album/artist]
 [?artist-name :eav ?artist :artist/name])


;; =>

;; It looks like with this query I never have to intersect...
;; Umm, when do I have to intersect?

(time
 (doall
  (for [?playlist (get-in indexes [:vae "Heavy Metal Classic" :playlist/name])
        ?track (get-in indexes [:eav ?playlist :playlist/track])
        ?track-name (get-in indexes [:eav ?track :track/name])
        ?album (get-in indexes [:eav ?track :track/album])
        ?album-name (get-in indexes [:eav ?album :album/name])
        ?artist (get-in indexes [:eav ?album :album/artist])
        ?artist-name (get-in indexes [:eav ?artist :artist/name])]
    {:track/name ?track-name
     :album/name ?album-name
     :artist/name ?artist-name})))
;; 1.7 ms :(
;; That is soooo slow :(





;; This is mixing up different kinds of indexes
;; We can deal with that fact later.


(def old-indexes
  {:playlist/name {"Heavy Metal Classic" [0 3]}
   :playlist-track/playlist {0 [0 1 2 3]
                             1 [0 1]
                             3 [2]}
   :playlist-track/track {0 [2]
                          1 [3]
                          2 [5]
                          3 [1]}
   :track/album {2 [3]}
   :album/artist {3 [4]}
   :artist/name {4 ["Death"]}})




[
 ]




;; Okay I think this is something like this.
;; Need to make some real proper indexes (maybe eav, etc) and try it out.
(time
 (for [playlist-id (intersect (get-in old-indexes [:playlist/name "Heavy Metal Classic"])
                              (into [] (keys (get-in old-indexes [:playlist-track/playlist]))))]
   (for [track-id (intersect (get-in old-indexes [:playlist-track/playlist playlist-id])
                             (into [] (keys (get-in indexes [:playlist-track/track]))))]
     [playlist-id track-id (get-in old-indexes [:playlist-track/track track-id])])))






(let [upper-1 (rand-int 1000000)
      lower-1 (rand-int upper-1)
      upper-2 (+ lower-1 (rand-int upper-1))
      lower-2 (rand-int upper-2)
      upper-3 (+ lower-2 (rand-int upper-2))
      lower-3 (rand-int upper-3)
      column-a (into [] (range lower-1 upper-1))
      column-b (into [] (range lower-2 upper-2))
      column-c (into [] (range lower-3 upper-3))]
  (println (+ (count column-a) (count column-b) (count column-c)))
  (time
   (do
     (println
      (count
       (intersect column-c
                  (intersect column-a column-b))))
     nil)))




