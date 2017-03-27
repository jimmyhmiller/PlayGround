(ns traffic.core
 (:use seesaw.core
        seesaw.graphics
        seesaw.color))


(def lane-width 20)
(def road-segment-length 20)
(def example-road (straight-road {:x 20 :y 20} 10 :y 4))


(defrecord car [length road lane speed pos heading])

(defrecord road [length path cars intersections lanes])

(defn each-lane [num-lanes dir x y]
  (if (= dir :y)
    (into {} (map (fn [i] [(keyword (str "lane-" i)) {:x (+ x (* lane-width i)) :y y}]) (range num-lanes)))
    (into {} (map (fn [i] [(keyword (str "lane-" i)) {:x x :y (+ y (* lane-width i))}]) (range num-lanes)))))

(each-lane 4 :x 20 20)

(each-lane)

(defn straight-road [{:keys [x y]} length dir lanes]
  (if (= dir :y)
    (into {} (map (fn [i] [(keyword (str "segment-" i)) (each-lane lanes dir x (+ y (* road-segment-length i)))]) (range length)))
    (into {} (map (fn [i] [(keyword (str "segment-" i)) (each-lane lanes dir (+ y (* road-segment-length i)) x)]) (range length)))))

(straight-road {:x 20 :y 20} 10 :x 4)






(def pos (ref {:x 1 :y 1}))

(dosync (ref-set pos {:x 20 :y 20}))


(defn draw-lane [g {:keys [x y]} edge]
  (doto g
    (.setColor (color :black))
    (.fillRect x y lane-width road-segment-length))
    (cond
     (= edge :left)
       (doto g
         (.setColor (color :white))
         (.fillRect x y 2 road-segment-length)
         (.fillRect (+ x (- lane-width 2)) y 2 road-segment-length)
         (.setColor (color :black))
         (.fillRect (+ x (- lane-width 2)) (+ y (* (/ 2 5) road-segment-length)) 2 (* (/ 1 5) road-segment-length)))
     (= edge :right)
       (doto g
         (.setColor (color :white))
         (.fillRect x y 2 road-segment-length)
         (.fillRect (+ x (- lane-width 2)) y 2 road-segment-length)
         (.setColor (color :black))
         (.fillRect x (+ y (* (/ 2 5) road-segment-length)) 2 (* (/ 1 5) road-segment-length)))
     (= edge :center-left)
       (doto g
         (.setColor (color :yellow))
         (.fillRect (+ x (- lane-width 2)) y 1 road-segment-length))
     (= edge :center-right)
       (doto g
         (.setColor (color :yellow))
         (.fillRect (+ x 2) y 1 road-segment-length))))

(defn draw-object [c g]
  (dosync
   (draw-road g example-road)))

(defn draw-road [g road]
  (doseq [[_ lanes] road [i lane] lanes] (draw-lane g lane ({:lane-0 :left  :lane-1 :center-left :lane-2 :center-right :lane-3 :right} i) )))





(defn -main []
  (-> (frame
    :title "Canvas Example"
    :width 500 :height 300
    :size [300 :by 400]
    :content
    (border-panel :hgap 5 :vgap 5 :border 5
                  ; Create the canvas with initial nil paint function, i.e. just canvas
                  ; will be filled with it's background color and that's it.
                  :center (canvas :id :canvas :background "#BBBBDD" :paint draw-object)))
  pack!
  show!))


(def prog (-main))
(-> (select prog [:#canvas])
    (repaint!))

