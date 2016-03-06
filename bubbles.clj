(defn in?
  "true if coll contains elm"
  [coll elm]
  (some #(= elm %) coll))

(def letters [["T" "H"]
              ["I" "S"]])


(defn get-letter [letters [x y] ]
  (-> letters
      (get x)
      (get y)))


'([0 0]
 ([1 0]
  ([1 1]
   ([0 1]))
  ([0 1]
   ([1 1])))
 ([0 1]
  ([1 1]
   ([1 0]))
  ([1 0]
   ([1 1])))
 ([1 1]
  ([0 1]
   ([1 0]))
  ([1 0]
   ([0 1]))))



(defn neighbors [[x y] size]
  [[(mod (+ x 1) size) y] [x (mod (+ y 1) size)] [(mod (+ x 1) size) (mod (+ y 1) size)]])


(defn get-paths [[x y] size exclusions]
  (let [neigh (filter (fn [path] (not (in? exclusions path))) (neighbors [x y] size))]
    (if (zero? (count neigh))
               (list [x y])
    (cons [x y] (map (fn [new-path] (get-paths new-path size (cons new-path exclusions))) neigh)))))

 (get-paths [0 0] 2 '([0 0]))

(defn get-words-for-depth
   ([letters path-tree depth]
    (get-words-for-depth letters path-tree depth 0))
   ([letters path-tree desired-depth depth]
    (if (= desired-depth depth)
      (list (get-letter letters (first path-tree)))
      (flatten (map (fn [path-branch]
                  (flatten (map (fn [letter] (str (get-letter letters (first path-tree)) letter))
                         (get-words-for-depth letters path-branch desired-depth (inc depth)))))
           (rest path-tree))))))


(get-words-for-depth letters (get-paths [0 0] 2 '([0 0])) 3)

(get-letter letters [0 1])

(defn get-neighbors [letters [x y]]
  (map (partial get-letter letters) (neighbors letters [x y])))


(defn get-next-words [letters [x y]]
  (map (fn [l] (str (get-letter letters [x y]) l)) (get-neighbors letters [x y])))