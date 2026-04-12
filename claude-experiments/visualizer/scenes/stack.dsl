;; === Stack Visualization ===
;; Everything reads from the active theme — palette, strokes, motion.

(scene stack
  (defc slot-width 140 :min 60 :max 240 :step 1 :category layout)
  (defc slot-height 42 :min 24 :max 80 :step 1 :category layout)
  (defc base-y 520 :min 200 :max 580 :step 1 :category layout)

  ;; State
  (def depth 0)
  (def values (list "alpha" "beta" "gamma" "delta"
                    "epsilon" "zeta" "eta" "theta"))

  ;; Base plate
  (rect :x 400 :y base-y
        :w (+ slot-width 40) :h 8
        :fill (stroke-color) :radius 2)
  (rect :x 400 :y (+ base-y 10)
        :w (+ slot-width 80) :h 3
        :fill (stroke-color) :radius 1)

  ;; Container walls
  (rect :x (- 400 (/ slot-width 2) 6)
        :y (- base-y 160)
        :w 4 :h 320
        :fill (stroke-color) :radius 1)
  (rect :x (+ 400 (/ slot-width 2) 6)
        :y (- base-y 160)
        :w 4 :h 320
        :fill (stroke-color) :radius 1)

  ;; Depth indicator
  (circle :x 200 :y 100 :r 6
          :fill (palette 5))

  ;; Click hint
  (circle :id "hint"
          :x 400 :y (- base-y 280) :r 8
          :fill (palette 4))

  ;; Click handler: push next value
  (on :click
    (when (< depth (len values))
      (sequence
        (let slot depth)

        (spawn! "_spawned"
          (rect :x (spring 400)
                :y (spring
                     (- base-y 6 (/ slot-height 2)
                        (* slot (+ slot-height 4)))
                     :initial 80)
                :w slot-width :h slot-height
                :radius 5
                :fill (palette slot)
                :scale (tween 0 1)))

        (set! depth (+ depth 1))))))
