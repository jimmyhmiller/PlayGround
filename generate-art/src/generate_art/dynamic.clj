(ns generate-art.dynamic
  (:require [quil.core :as q :include-macros true] 
            [quil.middleware :as m]))



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


(defn draw-trangle []
  (q/no-loop)
  (doseq [y (range 0 (q/height) 5)]
    (let [hue (rescale y 0 (q/height) 180 220)]
      (q/no-stroke)
      (q/fill hue 30 90)
      (q/rect 0 y (q/width) 15)))
  (doseq [y (range 0 (q/height) 10)
          x (range 0 (q/width) 10)]  
    (let [hue (rescale (random y (+ y 200)) 0 (q/height) 30 200)
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
    (let [hue (rescale (random y (+ y 200)) 0 (q/height) 0 360)
          brightness (random 50 100)]
      (q/stroke-weight (random 0 5))
      (q/stroke hue (random 50 70) brightness)
      (q/line 0 y (q/width) y)))
  (q/save "sketch.tif"))

(defn draw []
  (draw-richter))


(comment
  (global-set-key 
   (kbd "C-c C-s") 
   (lambda ()
           (interactive)
           (save-buffer)
           (cider-eval-defun-at-point) )))
