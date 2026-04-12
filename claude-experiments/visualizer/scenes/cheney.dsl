;; === Cheney's Copying GC ===
;; Semi-space collector: from-space on top, to-space below.
;; Click to step through the algorithm.

(scene cheney
  (defc cell-w 60 :min 30 :max 100 :step 1 :category layout)
  (defc cell-h 50 :min 30 :max 80 :step 1 :category layout)
  (defc gap 6 :min 0 :max 20 :step 1 :category layout)
  (defc from-y 160 :min 60 :max 300 :step 1 :category layout)
  (defc to-y 360 :min 200 :max 500 :step 1 :category layout)
  (defc left-x 120 :min 40 :max 300 :step 1 :category layout)

  ;; -- from-space cells (8 slots) --
  (each i (range 8)
    (let cx (+ left-x (* i (+ cell-w gap))))
    (rect :x cx :y from-y
          :w cell-w :h cell-h
          :fill (palette 5) :radius 6))

  ;; -- to-space cells (8 slots) --
  (each i (range 8)
    (let cx (+ left-x (* i (+ cell-w gap))))
    (rect :x cx :y to-y
          :w cell-w :h cell-h
          :fill (palette 6) :radius 6))

  ;; -- test line --
  (line :x1 100 :y1 100 :x2 300 :y2 100
        :color (palette 2) :stroke-width 2)

  ;; -- test arrow --
  (arrow :x1 100 :y1 500 :x2 300 :y2 500
         :color (palette 1) :stroke-width 2 :head-size 10)

  ;; -- root arrow pointing into from-space --
  (arrow :x1 50 :y1 80 :x2 120 :y2 (- from-y 30)
         :color (palette 0) :stroke-width 2.5 :head-size 10))
