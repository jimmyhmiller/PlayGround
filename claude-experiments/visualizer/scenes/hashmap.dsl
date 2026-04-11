;; === Interactive Hash Map ===
;; Click step to insert entries!

(scene hashmap
  ;; Tweakable constants (sliders in the panel)
  (defc bucket-width 70 :min 30 :max 120 :step 1 :category layout)
  (defc bucket-height 200 :min 80 :max 300 :step 1 :category layout)
  (defc bucket-y 350 :min 100 :max 500 :step 1 :category layout)
  (defc bucket-gap 16 :min 0 :max 40 :step 1 :category layout)
  (defc item-width 54 :min 20 :max 100 :step 1 :category layout)
  (defc item-height 36 :min 16 :max 60 :step 1 :category layout)
  (defc entry-stiffness 300 :min 50 :max 800 :step 10 :category springs)
  (defc entry-damping 18 :min 1 :max 40 :step 0.5 :category springs)

  ;; State
  (def op-index 0)
  (def arrow-x 400)
  (def arrow-alpha 0)
  (def entries (list))
  (def ops (list "name" "age" "city" "job"
                 "lang" "os" "ide" "db"))

  ;; Bucket array
  (each i (range 8)
    (let cx (+ 90 (* i (+ bucket-width bucket-gap))))
    (rect :x cx :y bucket-y
          :w bucket-width :h bucket-height
          :fill "#1a0e2a" :radius 12)
    (rect :x cx :y bucket-y
          :w (- bucket-width 4) :h (- bucket-height 4)
          :fill "#2a1848" :radius 10)
    (circle :x cx :y (- bucket-y 115)
            :r 5
            :fill (rgba 0.9 0.6 1.0 0.8)))

  ;; Hash arrow (tracks state)
  (triangle :id "arrow"
            :x (spring arrow-x
                       :stiffness 300
                       :damping 18)
            :y 248 :size 10
            :fill (rgba 1 0.8 0.2 0.9)
            :opacity (spring arrow-alpha
                             :stiffness 200
                             :damping 15))

  ;; Click hint
  (circle :id "hint"
          :x 400 :y 530 :r 8
          :fill (rgba 0.5 0.7 1 0.7))

  ;; Click handler
  (on :click
    (when (< op-index (len ops))
      (sequence
        ;; Compute target bucket
        (let key (nth ops op-index))
        (let bucket (hash key 8))
        (let slot (count-where entries
                               "bucket" bucket))

        ;; Show arrow — live-tracked via tweakables
        (set! arrow-x
              (+ 90 (* bucket
                       (+ bucket-width bucket-gap))))
        (set! arrow-alpha 1)
        (wait 0.4)

        ;; Spawn entry — springs from top, target follows tweakables
        (spawn! "_spawned"
          (rect :x (spring
                     (+ 90 (* bucket
                              (+ bucket-width
                                 bucket-gap)))
                     :initial 400
                     :stiffness entry-stiffness
                     :damping entry-damping)
                :y (spring
                     (- (+ bucket-y (/ bucket-height 2))
                        (+ (/ item-height 2)
                           8
                           (* slot
                              (+ item-height 4))))
                     :initial 80
                     :stiffness entry-stiffness
                     :damping entry-damping)
                :w item-width :h item-height :radius 6
                :fill (color-for op-index)
                :scale (tween 0 1
                              :duration 0.4
                              :easing back-out)))

        ;; Record the entry
        (push! entries
          (:bucket bucket :slot slot))
        (set! op-index (+ op-index 1))

        (wait 0.5)

        ;; Hide arrow
        (set! arrow-alpha 0)))))
