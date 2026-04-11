;; === Stack Visualization ===
;; Push values onto a stack, one click at a time

(scene stack
  (defc slot-width 140 :min 60 :max 240 :step 1 :category layout)
  (defc slot-height 42 :min 24 :max 80 :step 1 :category layout)
  (defc base-y 520 :min 200 :max 580 :step 1 :category layout)
  (defc push-stiffness 250 :min 50 :max 600 :step 10 :category springs)
  (defc push-damping 16 :min 1 :max 40 :step 0.5 :category springs)

  ;; State
  (def depth 0)
  (def values (list "alpha" "beta" "gamma" "delta"
                    "epsilon" "zeta" "eta" "theta"))

  ;; Base plate
  (rect :x 400 :y base-y
        :w (+ slot-width 40) :h 8
        :fill "#3a3d4a" :radius 2)
  (rect :x 400 :y (+ base-y 10)
        :w (+ slot-width 80) :h 3
        :fill "#2a2d38" :radius 1)

  ;; Container walls (visual only)
  (rect :x (- 400 (/ slot-width 2) 6)
        :y (- base-y 160)
        :w 4 :h 320
        :fill "#2a2d38" :radius 1)
  (rect :x (+ 400 (/ slot-width 2) 6)
        :y (- base-y 160)
        :w 4 :h 320
        :fill "#2a2d38" :radius 1)

  ;; Depth indicator
  (circle :x 200 :y 100 :r 6
          :fill (rgba 0.5 0.7 1 0.8))

  ;; Click hint
  (circle :id "hint"
          :x 400 :y (- base-y 280) :r 8
          :fill (rgba 0.5 0.7 1 0.6))

  ;; Click handler: push next value
  (on :click
    (when (< depth (len values))
      (sequence
        (let slot depth)

        ;; Spawn new stack item at top, springing into place
        (spawn! "_spawned"
          (rect :x (spring 400
                           :stiffness push-stiffness
                           :damping push-damping)
                :y (spring
                     (- base-y 6 (/ slot-height 2)
                        (* slot (+ slot-height 4)))
                     :initial 80
                     :stiffness push-stiffness
                     :damping push-damping)
                :w slot-width :h slot-height
                :radius 5
                :fill (color-for slot)
                :scale (tween 0 1
                              :duration 0.35
                              :easing back-out)))

        (set! depth (+ depth 1))))))
