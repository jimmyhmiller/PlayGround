(ns testing-stuff.grid)

(defn index [x y max-columns]
  (+ x (* y max-columns)))

(index 0 0 2)
(index 1 0 2)
(index 0 1 2)
(index 1 1 2)

(defn cells->indexes [max-columns {:keys [x y w h]}]
  (sort 
   (for [x (range x (+ x w))
         y (range y (+ y h))]
     (index x y max-columns))))


(cells->indexes 2 {:x 0 :y 0 :w 2 :h 2})

(def cells [{:x 0 :y 0 :w 1 :h 1} {:x 1 :y 0 :w 1 :h 1} {:x 1 :y 1 :w 1 :h 1}])


(map (fn [x y] [x y]) (range 0 3) (range 0 3))

(for [x (range 0 2)
      y (range 0 2)]
  (index x y 2))
(sort '(0 2 1 3))
