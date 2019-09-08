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
  (try
    (+ mean (* variance (q/random-gaussian)))
    (catch Exception e 0.5)))




(defn random-point [[x1 y1] [x2 y2]]
  (let [n (gauss 0.5 0.1)] 
    [(+ (* n x1) (* (- 1 n) x2))
     (+ (* n  y1) (* (- 1 n) y2))])
  #_(let [denom 4
        num (q/random 1 4)
        diff (- denom num)]
    [(+ (* (/ num denom) x1) (* (/ diff denom) x2))
     (+ (* (/ num denom) y1) (* (/ diff denom) y2))]))


(defn random-point' [[x1 y1] [x2 y2]]
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

(defn random 
  ([x]
   (q/random x))
  ([x y]
   (try 
     (q/random x y)
     (catch Exception e (+ x (rand-int (- (inc y) x)))))))

(defn odds [n]
  (try
    (< (q/random 1) n)
    (catch Exception e false)))

(defn subdivide-r [[p1 p2 p3]]
  (let [triangles
        (second (max-key (comp distance first) 
                         [[p1 p2] (let [r (random-point p1 p2)] [[r p2 p3] [r p1 p3]])]
                         [[p2 p3] (let [r (random-point p2 p3)] [[r p3 p1] [r p2 p1]])]
                         [[p1 p3] (let [r (random-point p1 p3)] [[r p3 p2] [r p1 p2]])]))]
    ;triangles
    (cond
      (odds 0.005) (take 1 triangles)
      (odds 0.005) (drop 1 triangles)
      :else triangles)))

(defn subdivide-r-lossless [[p1 p2 p3]]
  (let [triangles
        (second (max-key (comp distance first) 
                         [[p1 p2] (let [r (random-point' p1 p2)] [[r p2 p3] [r p1 p3]])]
                         [[p2 p3] (let [r (random-point' p2 p3)] [[r p3 p1] [r p2 p1]])]
                         [[p1 p3] (let [r (random-point' p1 p3)] [[r p3 p2] [r p1 p2]])]))]
    triangles))




(defn subdivide [[p1 p2 p3]]
  (second (max-key (comp distance first) 
                   [[p1 p2] [[(midpoint p1 p2) p2 p3] [(midpoint p1 p2) p1 p3]]]
                   [[p2 p3] [[(midpoint p2 p3) p3 p1] [(midpoint p2 p3) p2 p1]]]
                   [[p1 p3] [[(midpoint p1 p3) p3 p2] [(midpoint p1 p3) p1 p2]]])))

(defn tri [[[x1 y1] [x2 y2] [x3 y3]]]
  (q/triangle x1 y1 x2 y2 x3 y3))



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


(defn height []
  (try (q/height)
       (catch Exception e 1000)))

(defn width []
  (try (q/width)
       (catch Exception e 1000)))

(defn initial-triangles' []
  [[[(width) 0] 
    [0 (* 1 (/ (height) 3))] 
    [0 0]] 
   [[(width) 0] 
    [0 (* 1 (/ (height) 3))] 
    [(width) (* 1 (/ (height) 3))]] 
   [[(width) (* 1 (/ (height) 3))] 
    [(width) (* 2 (/ (height) 3))] 
    [0 (* 1 (/ (height) 3))]]
   [[0 (* 1 (/ (height) 3))] 
    [0 (* 2 (/ (height) 3))] 
    [(width) (* 2 (/ (height) 3))]]
   [[0 (* 2 (/ (height) 3))] 
    [0 (height)] 
    [(width) (* 2 (/ (height) 3))]]
   [[0 (height)]
    [(width) (height)] 
    [(width) (* 2 (/ (height) 3))]]])


(defn power-2 [x]
  (Math/pow 2 x))

(defn power-10 [x]
  (Math/pow 10 x))

(defn triangles []
  (mapcat (fn [tri n] (take n (generate-triangles [tri] subdivide-r-lossless)))
       (initial-triangles')
       [(power-10 2) (power-10 4)
        (power-10 6) (power-10 8)
        (power-10 10) (power-10 12)]))



(defn t1 [] [(nth 1 (initial-triangles'))])
(defn t2 [] [(nth 2 (initial-triangles'))])
(defn t3 [] [(nth 3 (initial-triangles'))])
(defn t4 [] [(nth 4 (initial-triangles'))])
(defn t5 [] [(nth 5 (initial-triangles'))])
(defn t6 [] [(nth 6 (initial-triangles'))])



(defn clear-canvas []
  (q/no-loop)
  (q/background 360))


(defn r+- [x y]
  (+ x (random (- y) y)))

(defn r- [x y]
  (+ x (random (- y) 0)))

(range 0 1 10)

(defn random-step-range [start stop step-low step-high]
  (let [step (random step-low step-high)]
    (cond 
      (= start stop) '()
      (>= (+ start step) stop) (list stop)
      :else (cons start (random-step-range (+ start step) stop step-low step-high)))))

(defn step-with-length [start stop low high]
  (let [ranger (random-step-range start stop low high)]
    (map (fn [x y] [x (- y x)]) ranger (rest ranger))))




(defn draw []
  (clear-canvas)
  (q/stroke-weight 1)
  (doseq [y (random-step-range 0 (height) 0 10)]
    (doseq [[x length] (step-with-length 0 (width) 10 10)]
      (q/curve (r+- x 10) (r+- y 10)
               x y (+ x length) y
               (r+- x 10) (r+- y 10)))))




(defn slope [[x1 y1] [x2 y2]]
  (/ (- y2 y1)
     (- x2 x1)))

(defn diagonal [width height x]
  (int (* x (slope [0 0] [width height]))))


(def position (atom 200))

(defn draw []
  (q/background 360)
  (q/no-loop)
  (q/line 10
          100
          100
          @position))



(defn key-release []
  (when (and (= (q/raw-key) \i))
    (swap! key-meta assoc \i nil)
    (q/redraw)))

(def key-meta (atom {}))


;; Idea, be able to control randomness via keyboard


(defn key-press []
  (prn (q/raw-key))
  (when (and (= (q/raw-key) \i) (not (get @key-meta \i)))
    (swap! key-meta assoc \i {:pressed true 
                              :mouse-y (q/mouse-y)})
    (q/redraw))
  (when (= (q/raw-key) \r)
    (q/redraw)))



(def iterations (atom 1))


(defn draw []
  (clear-canvas)
  (q/no-fill)
  (q/stroke-weight 0.5)

  (q/no-stroke)
  (q/fill 0)
  (q/stroke 0 0 0)
  (doseq [i (range @iterations)]
    (doseq [x (range 0 (width))]
      (let [y (diagonal (width) (height) x) 
            offset (gauss 10 600)
            offset2 (gauss 10 600)]
        (when (odds 0.9)
          (q/line (+ x 0) 
                  (- offset offset) 
                  (+ x offset)
                  (+ y 0))))))
  (q/rect 600 800 200 100)
  (q/fill 360)
  (q/text-size 20)
  (q/text (str  (get-in @key-meta [\i :mouse-y]))
          620 850)
  

  (q/save "sketch.tif"))


(defn draw []
  (clear-canvas)
  (q/fill 0 0 100)
  (q/rect 0 0 (q/width) (q/height))
  (q/no-loop)
  (q/no-fill)
  (q/stroke-weight 0.5)
  (q/stroke 0 0 0)
  (doseq [i (range 2)]
    (doseq [x (range 0 (width) 5)
            y (range 0 (height))
            :when (= y 1)]

     #_ (when (odds 0.02)
        (doseq [_ (range 10)]
          (let [offset (gauss 0 10)]
            (when (odds 0.9)
              (q/line (- offset offset) 
                      (+ y offset) 
                      x 
                      (+ y offset))))))
      #_(let [offset (gauss 0 10)]
        (when (odds 0.9)
          (q/line (- offset offset) 
                  (+ y offset) 
                  x 
                  (+ y offset))))

      #_(when (odds 0.02)
        (doseq [_ (range 10)]
          (let [offset (gauss 0 10)]
            (q/line (+ x offset) 
                    (- offset offset) 
                    (+ x offset)
                    y))))
      (let [offset (gauss 10 10)]
        (when (odds 0.9)
          (q/curve (+ x offset) 
                   (- offset (r+ offset 20)) 
                   (+ (r+- x 100) (r+- offset 100))
                   (r+ y 10000)
                   (+ x offset) 
                   (- offset (r+ x 20)) 
                   (+ (r+- x 100) (r+- offset 100))
                   (r+ x y))))))
  #_(doseq [x (range 0 (width) 5)
          y (range 0 (height))]
    (let [offset (random 0 3)]
      (when (odds 0.1)
        (q/line (- x 10) (+ y offset) x (+ y offset))))
    (let [offset (random 0 3)]
      (when (odds 0.1)
        (q/line (+ x offset) 
                (- y 10) 
                (+ x offset)
                y))))
  (q/save "sketch.tif"))


(defn draw []
  (clear-canvas)
  (q/fill 0 0 100)
  (q/rect 0 0 (q/width) (q/height))
  (q/no-loop)
  (q/no-fill)
  (q/stroke-weight 0.5)
  (q/stroke 0 0 0)
  (doseq [i (range 10)]
    (doseq [x (range 0 (width) 5)
            y (range 0 (height))
            :when (= (diagonal (width) (height) x) y)]

      (when (odds 0.02)
        (doseq [_ (range 10)]
          (let [offset (gauss 0 10)]
            (when (odds 0.9)
              (q/line (- offset offset) 
                      (+ y offset) 
                      x 
                      (+ y offset))))))
      (let [offset (gauss 0 10)]
        (when (odds 0.9)
          (q/line (- offset offset) 
                  (+ y offset) 
                  x 
                  (+ y offset))))

      (when (odds 0.02)
        (doseq [_ (range 10)]
          (let [offset (gauss 0 10)]
            (q/line (+ x offset) 
                    (- offset offset) 
                    (+ x offset)
                    y))))
      (let [offset (gauss 10 10)]
        (when (odds 0.9)
          (q/line (+ x offset) 
                  (- offset offset) 
                  (+ x offset)
                  y)))))
  (doseq [x (range 0 (width) 5)
          y (range 0 (height))]
    (let [offset (random 0 3)]
      (when (odds 0.1)
        (q/line (- x 10) (+ y offset) x (+ y offset))))
    (let [offset (random 0 3)]
      (when (odds 0.1)
        (q/line (+ x offset) 
                (- y 10) 
                (+ x offset)
                y))))
  (q/save "sketch.tif"))


(defn draw []
  (clear-canvas)
  (let [tris (triangles)]
    (q/no-loop)
    ;(q/fill 49 9 96)
    (q/fill 0 0 100)
    (q/rect 0 0 (q/width) (q/height))
    (q/no-fill)
    (q/stroke-weight 0.5)
    (q/stroke 0 0 0)
    
    (doseq [t tris]
      (tri t))
    (q/save "sketch.tif")))


(def rotations [q/PI (/ q/PI 4) (/ q/PI 2) (* 2 (/ q/PI 2))])

(defn right-triangle 
  ([x y width height]
   (right-triangle x y width height 0))
  ([x y width height rotate]
   (q/triangle
    (+ x width) (+ y width)
    (+ x 0) (+ y 0)
    (+ x height) (+ y 0))))

(defn draw []
  (q/no-loop)
  (clear-canvas)
  (doseq [y (range 0 (q/height) 5)]
    (let [hue (rescale y 0 (q/height) 180 220)]
      (q/no-stroke)
      (q/fill hue 30 90)
      (q/rect 0 y (q/width) 15)))
  (doseq [y (range 0 (q/height) 10)
          x (range 0 (q/width) 10)]
    (let [hue (rescale (q/random y (+ y 1000)) 0 (q/height) 30 220)
          size-0 (gauss 1 50)
          size-1 (gauss 1 10)
          size-2 (gauss 1 5)
          size-3 (gauss 1 2)
          rotate (rand-int 2)
          offset-x 0
          offset-y 0
          to-draw (q/random 0 10)]
      (q/no-stroke)
      (q/fill hue 30 90)
      (when (< (r+ 1 300) to-draw)
        (right-triangle (+ offset-x x) (+ offset-y y) size-0 size-0 rotate))
      (when (< (r+ 1 4) to-draw)
        (right-triangle (+ offset-x x) (+ offset-y y) size-1 size-1 rotate))
      (when (< (r+ 1 4) to-draw)
        (right-triangle (+ offset-x x) (+ offset-y y) size-2 size-2 rotate))
      (when (< (r+ 1 4) to-draw)
        (right-triangle (+ offset-x x) (+ offset-y y) size-3 size-3 rotate))))
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


(defn draw []
  (q/no-loop)
  (q/background 0 0 100)
  (doseq [y (range 0 (q/height) 1)]
    (let [hue (rescale (q/random y (+ y 500)) 0 (q/height) 0 360)
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
