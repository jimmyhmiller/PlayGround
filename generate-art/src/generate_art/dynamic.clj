(ns generate-art.dynamic
  (:require [quil.core :as q :include-macros true] 
            [quil.middleware :as m]
            [roul.random :as rr]))



(defn setup []
  (q/color-mode :hsb 360 100 100 1.0))


(defn rescale [value old-min old-max new-min new-max]
  "Rescales value from range [old-min, old-max] to [new-min, new-max]"
  (let [old-spread (- old-max old-min)
        new-spread (- new-max new-min)]
    (+ (* (- value old-min) (/ new-spread old-spread))
       new-min)))

(defn r+ [x y]
  (+ x (q/random 0 y)))


(defn square [x]
  (* x x))

(defn distance [[[x1 y1] [x2 y2]]]
   (Math/sqrt (+ (square (- x2 x1)) 
                   (square (- y2 y1)))))
(defn gauss [mean variance]
  (+ mean (* variance (q/random-gaussian))))




(defn random-point [[x1 y1] [x2 y2]]
  (let [n (gauss 0.5 0.1)] 
    [(+ (* n x1) (* (- 1 n) x2))
     (+ (* n  y1) (* (- 1 n) y2))])
  #_(let [denom 4
        num (q/random 1 4)
        diff (- denom num)]
    [(+ (* (/ num denom) x1) (* (/ diff denom) x2))
     (+ (* (/ num denom) y1) (* (/ diff denom) y2))]))



(defn midpoint [[x1 y1] [x2 y2]] 
  [(* (/ 1 2) (+ x1 x2))
   (* (/ 1 2) (+ y1 y2))])


(defn odds [n]
  (< (q/random 1) n))

(defn subdivide-r [[p1 p2 p3]]
  (let [triangles
        (second (max-key (comp distance first) 
                         [[p1 p2] (let [r (random-point p1 p2)] [[r p2 p3] [r p1 p3]])]
                         [[p2 p3] (let [r (random-point p2 p3)] [[r p3 p1] [r p2 p1]])]
                         [[p1 p3] (let [r (random-point p1 p3)] [[r p3 p2] [r p1 p2]])]))]
    triangles
    (cond
      (odds 0.05) (take 1 triangles)
      (odds 0.05) (drop 1 triangles)
      :else triangles)))




(defn subdivide [[p1 p2 p3]]
  (second (max-key (comp distance first) 
                   [[p1 p2] [[(midpoint p1 p2) p2 p3] [(midpoint p1 p2) p1 p3]]]
                   [[p2 p3] [[(midpoint p2 p3) p3 p1] [(midpoint p2 p3) p2 p1]]]
                   [[p1 p3] [[(midpoint p1 p3) p3 p2] [(midpoint p1 p3) p1 p2]]])))

(defn tri [[[x1 y1] [x2 y2] [x3 y3]]]
  (q/triangle x1 y1 x2 y2 x3 y3))


(subdivide [[0 0] [0 1000] [1000 0]])


(defn generate-triangles 
  ([initials]
   (generate-triangles initials subdivide))
  ([initials subdivide]
   (->> initials
        (iterate (partial mapcat subdivide)) 
        (mapcat identity))))


(defn initial-triangles []
  [[[(q/width) (q/height)] [0 (q/height)] [(q/width) 0]]
   [[0 0] [0 (q/height)] [(q/width) 0]]])


(defn initial-triangles' []
  [[[0 (q/height)] [(q/width) (q/height)] [(q/width) (* 2 (/ (q/height) 3))]]
   [[0 (* 1 (/ (q/height) 3))] 
    [0 (* 2 (/ (q/height) 3))] 
    [(q/width) (* 2 (/ (q/height) 3))]]
   [[(q/width) 0] 
    [0 (* 1 (/ (q/height) 3))] 
    [(q/width) (* 1 (/ (q/height) 3))]]
   [[(q/width) 0] 
    [0 (* 1 (/ (q/height) 3))] 
    [0 0]]])


(defn draw []
  (let [triangles (take 20000 (generate-triangles (initial-triangles') subdivide-r))]
    (q/no-loop)
    ;(q/fill 49 9 96)
    (q/fill 0 0 100)
    (q/rect 0 0 (q/width) (q/height))
    (q/no-fill)
    (q/stroke-weight 0.5)
    (q/stroke 0 0 0) 
    
    (doseq [t triangles]
      (tri t))
    (q/save "sketch.tif")))


(defn draw' []
  (q/no-loop)
  (doseq [y (range 0 (q/height) 5)]
    (let [hue (rescale y 0 (q/height) 180 220)]
      (q/no-stroke)
      (q/fill hue 30 90)
      (q/rect 0 y (q/width) 15)))
  (doseq [y (range -20 (q/height) 30)
          x (range -20 (q/width) 30)]
    (let [hue (rescale (q/random y (+ y 200)) 0 (q/height) 30 200)
          size (q/random 0 50)
          to-draw (q/random 0 10)]
      (q/no-stroke)
      (q/fill hue 30 90) 
      (when (< (r+ 1 4) to-draw)
        (q/triangle 
         (+ x 50) (+ y 50)
         (+ x 20) (+ y 20)
         (+ x 50) (+ y 20)))))
  (q/save "sketch.tif"))


(defn draw-triangle []
  (q/no-loop)
  (doseq [y (range 0 (q/height) 5)]
    (let [hue (rescale y 0 (q/height) 180 220)]
      (q/no-stroke)
      (q/fill hue 30 90)
      (q/rect 0 y (q/width) 15)))
  (doseq [y (range 0 (q/height) 10)
          x (range 0 (q/width) 5)]  
    (let [hue (rescale (q/random y (+ y 200)) 0 (q/height) 30 200)
          size (q/random 0 50)
          to-draw (q/random 0 (+ (q/height) 10))]
      (q/no-stroke)
      (q/fill hue 30 90)
      (when (> (+ to-draw (/ y 4)) (q/height))
        (q/triangle 
         (r+ x 50) (r+ y 50)
         (r+ x 20) y
         (r+ x 50) (r+ y 50)))))
  (q/save "sketch.tif"))


(defn draw-richter []
  (q/no-loop)
  (q/background 0 0 100)
  (doseq [y (range 0 (q/height) 1)]
    (let [hue (rescale (q/random y (+ y 200)) 0 (q/height) 0 360)
          brightness (q/random 50 100)]
      (q/stroke-weight (q/random 0 5))
      (q/stroke hue (q/random 50 70) brightness)
      (q/line 0 y (q/width) y)))
  (q/save "sketch.tif"))

#_(defn draw []
  (draw-richter))


(comment
  (global-set-key 
   (kbd "C-c C-s") 
   (lambda ()
           (interactive)
           (save-buffer)
           (cider-eval-defun-at-point) )))
