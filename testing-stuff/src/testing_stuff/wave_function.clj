(ns testing-stuff.wave-function)

(do
  (def image-data [[1 0 0 1]
                   [1 1 0 1]
                   [0 0 0 1]
                   [0 0 0 1]])


  (defn array-2d-to-flattened-data [image-data]
    (let [height (count image-data)
          width (count (first image-data))]
      {:coordinates
       (into {}
             (for [i (range height)
                   j (range width)]
               [[i j] (get-in image-data [i j])]))}))


  (def flattened-image (array-2d-to-flattened-data image-data)))



(def block "X")
(def empty "O")
(def draw-info [empty block])

(defn draw [height width image-data]
  (dotimes [i height]
    (println)
    (dotimes [j width]
      (print (get draw-info (get-in image-data [:coordinates [i j]])))))
  (println))


(defn rotate [[[y x] data] [height width] degrees]
  (case degrees
    90 [[x (- width y 1)] data]
    180 [[(- height y 1) (- width x 1)] data]
    270 [[(- height x 1) y] data]))


(defn rotate-image [width height degrees image]
  (assoc image :coordinates
         (reduce-kv (fn [acc k v]
                      (let [[k' v'] (rotate [k v] [height width] degrees)]
                        (assoc acc k' v')))
                    {}
                    (:coordinates image))))

(defn indices-for-area [n m offset-n offset-m]
  (for [i (range n)
        j (range m)]
    [[i j] [(+ offset-n i) (+ offset-m j)]]))

(defn get-area [image n m offset-n offset-m]
  (assoc image :coordinates
   (let [indices (indices-for-area n m offset-n offset-m)]
     (into {}
           (map (fn [[index index-in-image]]
                  [index (get-in image [:coordinates index-in-image])]) indices)))))

(defn get-overlapping-patterns [width height pattern-size image]
  (for [i (range (- height (dec pattern-size)))
        j (range (- width (dec pattern-size)))]
    (get-area image pattern-size pattern-size i j)))


(defn get-all-patterns [width height pattern-size image]
  (mapcat (fn [image]
            [image 
             (rotate-image pattern-size pattern-size 90 image)
             (rotate-image pattern-size pattern-size 180 image)
             (rotate-image pattern-size pattern-size 270 image)])
   (get-overlapping-patterns width height pattern-size image)))

(defn eliminate-duplicates [patterns]
  (reduce (fn [acc [image frequency]]
            (conj acc (assoc image :frequency frequency)))
          []
          (frequencies patterns)))

(defn get-row [row-n pattern-size pattern]
  (mapv (fn [x] (get-in pattern [:coordinates [row-n x]])) (range pattern-size)))

(defn get-col [col-n pattern-size pattern]
  (mapv (fn [y] (get-in pattern [:coordinates [y col-n]])) (range pattern-size)))


;; TODO: Move pattern size to pattern
(defn find-overlap [pattern-size pattern-1 pattern-2]
  (set
   (map first
        (filter second
                (let [size (dec pattern-size)]
                  [[:northwest (= (get-in pattern-1 [:coordinates [0 0]]) 
                                  (get-in pattern-2 [:coordinates [size size]]))]
                   [:north     (= (get-row size pattern-size pattern-1)
                                  (get-row 0 pattern-size pattern-2))]
                   [:northeast (= (get-in pattern-1 [:coordinates [0 size]]) 
                                  (get-in pattern-2 [:coordinates [size 0]]))]
                   [:east       (= (get-col size pattern-size pattern-1)
                                   (get-col 0 pattern-size pattern-2))]
                   [:southeast (= (get-in pattern-1 [:coordinates [size size]]) 
                                  (get-in pattern-2 [:coordinates [0 0]]))]
                   [:south      (= (get-row 0 pattern-size pattern-1)
                                   (get-row size pattern-size pattern-2))]
                   [:southwest (= (get-in pattern-1 [:coordinates [size 0]]) 
                                  (get-in pattern-2 [:coordinates [0 size]]))]
                   [:west      (= (get-col 0 pattern-size pattern-1)
                                  (get-col size pattern-size pattern-2))]])))))



(defn add-ids-to-patterns [patterns]
  (mapv (fn [pattern id] (assoc pattern :id id))
       patterns
       (range)))

(defn find-all-overlapping-cells-by-id [pattern-size patterns]
  (reduce (fn [acc [p1 p2]]
            (-> acc
                (update-in [(:id p1)]
                           (fn [overlaps]
                             (reduce (fn [acc direction] 
                                       (if (empty? (get acc direction))
                                         (assoc acc direction #{(:id p2)})
                                         (update acc direction conj (:id p2))))
                                     overlaps
                                     (find-overlap pattern-size p1 p2))))))
          {}
          (for [p1 patterns
                p2 patterns]
            [p1 p2])))

(defn add-overlapping-to-patterns [pattern-size patterns]
  (let [overlaps (find-all-overlapping-cells-by-id 2 patterns)]
    (mapv (fn [x] (assoc x :overlapping (overlaps (:id x)))) 
          patterns)))


;; TODO: Get rid of width height and instead put it in the image
;; TODO: Flip x and y
(defn get-patterns [width height pattern-size image]
  (->> (get-all-patterns width height pattern-size image)
       eliminate-duplicates
       add-ids-to-patterns
       (add-overlapping-to-patterns pattern-size)))


(defn in-bounds? [width height [y x]]
  (and (<= 0 y)
       (< y height)
       (<= 0 x)
       (< x width)))


;; This is wrong if pattern-size > 2
;; Overlap can be much more than this.

(defn neighbors [width height [y x]]
  (if-not (in-bounds? width height [y x])
    []
    (filterv (comp (partial (comp in-bounds?) width height) second)
             [[:northwest [(dec y) (dec x)]]
              [:north     [(dec y) x]]
              [:northeast [(dec y) (inc x)]]
              [:east      [y       (inc x)]]
              [:southeast [(inc y) (inc x)]]
              [:south     [(inc y)  x]]
              [:southwest [(inc y)  (dec x)]]
              [:west      [y        (dec x)]]])))



;; I'm thinking that a cell should just be a position.
;; Then we can say what patterns it has in the board.
;; we can say for each pattern what it can overlap with.
;; We can have patterns by id and anytime we are talking about
;; patterns, we just have a set.
;; We can also keep an eligible neighbors list per position.

(defn get-neighbors2 [board [coords cell]]
  (mapv (fn [[direction [y x]]] 
          {:direction direction
           :coords [y x]
           :patterns (set (map :id (:patterns (get-in board [:coordinates [y x]]))))})
        (:neighbors cell)))

(defn add-extra-neighbors [board]
  (update board :coordinates
          #(into 
            {} 
            (map
             (fn [[coords cell]]
               [coords (assoc cell :neighbors2 
                              (get-neighbors2 board [coords cell]))])
             %))))



(defn initialize-board [width height patterns]
  {:coordinates
   (into {}
         (for [i (range height)
               j (range width)]
           [[i j] {:patterns patterns
                   :entropy (reduce + (map :frequency patterns))
                   :neighbors (neighbors width height [i j])}]))})





;; TODO: use a heap
(defn lowest-entropy [board]
  (let [candidates (->> (:coordinates board)
                        (remove (fn [[k v]] (= (count (:patterns v)) 1)))
                        (sort-by (comp :entropy second))
                        #_(drop-while (comp zero? :entropy second)))]
    (if (empty? candidates)
      nil
      (rand-nth
       (into []
             (take-while (fn [x] (= (:entropy (second (first candidates)))
                                    (:entropy (second x))))
                         candidates))))))


(defn choose-pattern [{:keys [patterns entropy] :as cell}]
  (let [random-chance (* (rand) entropy)]
    (reduce (fn [chance pattern]
              (let [remaining-chance (- chance (:frequency pattern)) ]
                (if (<= remaining-chance 0)
                  (reduced pattern)
                  remaining-chance)))
            random-chance
            patterns)))

(defn collapse [[[y x] {:keys [patterns entropy] :as cell}]]
  (let [pattern (choose-pattern cell)]
    [[y x]
     (assoc cell
            :entropy (:frequency pattern)
            :patterns [pattern]
            :collapsed true)]))


(defn get-neighbors [board [coords cell]]
  (filterv (fn [[coords neighbor]]
             (> (count (:patterns neighbor)) 1))
           (mapv (fn [[direction [y x]]] 
                   [[y x] (get-in board [:coordinates [y x]])]) (:neighbors cell))))



(defn propagate-single* [board [coords cell]]
  )


(defn propagate-single [board [coords cell]]

  (let [neighbors (map (fn [[direction [y x]]] 
                         [direction [y x] (get-in board [:coordinates [y x]])]) (:neighbors cell))
        patterns (filterv (fn [pattern]
                            (every? false?
                                    (map (fn [[direction _ neighbor]]
                                           (let [possible-patterns (set (map :id (:patterns neighbor)))]
                                             (zero? (count (filter possible-patterns
                                                                   (get-in pattern [:overlapping direction]))))))
                                         neighbors))) 
                          (:patterns cell))]
      
    {:board
     (assoc-in board [:coordinates coords] (assoc cell 
                                                  :patterns patterns
                                                  :entropy (reduce + 0 (map :frequency patterns))))
     :flagged (if (not= (count (:patterns cell)) (count patterns))
                (do
                  (let [result (filterv
                                (fn [[coords neighbor]] (> (count (:patterns neighbor)) 1))
                                (mapv (fn [[direction coords neighbor]] [coords neighbor]) neighbors))]
                    result)
                  )
                [])}
    ))

(defn propagate [board flagged]
  (if (empty? flagged)
    board
    (let [results (propagate-single board (first flagged))]
      (if (:info results)
        results
        ;; order here is important, don't know why.
        (recur (:board results) (into [] (concat (:flagged results) (rest flagged) )))))))





(defn run-wave-collapse [board i]
  (if (> i 7000)
    (do (println "FAILLLL")
        board)
    (let [new-collapse (lowest-entropy board)]
      (cond  
        (not new-collapse)
        board
        (zero? (:entropy (second new-collapse)))
        (do (println "ZERO!")
            board)
        :else (do
                (let [[coord cell :as collapsed] (collapse new-collapse)]
                 
                  (recur (propagate (assoc-in board [:coordinates coord] cell) 
                                    (get-neighbors board collapsed))
                         (inc i))))))))
(comment)

(def board (initialize-board 100 100 (get-patterns 4 4 2 flattened-image)))

(type (:coordinates board))

(first  (:coordinates board))

(do
  (map :entropy
       (map second
            (:coordinates
             (run-wave-collapse board 0))))
  nil)



(draw 20 20 completed)
(draw 100 100
      {:coordinates
       (into {}
             (map (juxt first (comp #(get % [0 0]) :coordinates first :patterns second))
                  (:coordinates completed)))})

(def completed (run-wave-collapse board 0))

(:info (run-wave-collapse board 0))
(println "\n\n\n")


(draw 2 2 {:coordinates {[1 0] 0, [0 0] 1, [1 1] 1, [0 1] 1},
  :frequency 4,
  :id 3,
  :overlapping
  {:south #{0 1 4},
   :northwest #{0 1 4 3},
   :northeast #{0 1 4 2},
   :north #{0},
   :southwest #{1},
   :east #{1 4 2},
   :southeast #{1 4 3 2},
   :west #{2}}})


;; Add neighbors
;; Filter out patterns that don't belong
;; (We could really just give each pattern an id and then precompute overlaps and store in a set)




