;; === Interactive Hash Map ===
;; Fully theme-driven: palette colors, theme springs, theme tweens.

(scene hashmap
  (defc bucket-width 70 :min 30 :max 120 :step 1 :category layout)
  (defc bucket-height 200 :min 80 :max 300 :step 1 :category layout)
  (defc bucket-y 350 :min 100 :max 500 :step 1 :category layout)
  (defc bucket-gap 16 :min 0 :max 40 :step 1 :category layout)
  (defc item-width 54 :min 20 :max 100 :step 1 :category layout)
  (defc item-height 36 :min 16 :max 60 :step 1 :category layout)

  ;; State
  (def op-index 0)
  (def arrow-x 400)
  (def arrow-alpha 0)
  (def entries (list))
  (def ops (list "name" "age" "city" "job"
                 "lang" "os" "ide" "db"))

  ;; Bucket array — fills from palette slot 5 (slate blue in iso50),
  ;; strokes come from theme automatically.
  (each i (range 8)
    (let cx (+ 90 (* i (+ bucket-width bucket-gap))))
    (rect :x cx :y bucket-y
          :w bucket-width :h bucket-height
          :fill (palette 5) :radius 12)
    (circle :x cx :y (- bucket-y 115)
            :r 5
            :fill (palette 2)))

  ;; Hash arrow
  (triangle :id "arrow"
            :x (spring arrow-x)
            :y 248 :size 10
            :fill (palette 1)
            :opacity (spring arrow-alpha))

  ;; Click hint
  (circle :id "hint"
          :x 400 :y 530 :r 8
          :fill (palette 4))

  ;; Click handler
  (on :click
    (when (< op-index (len ops))
      (sequence
        (let key (nth ops op-index))
        (let bucket (hash key 8))
        (let slot (count-where entries
                               "bucket" bucket))

        (set! arrow-x
              (+ 90 (* bucket
                       (+ bucket-width bucket-gap))))
        (set! arrow-alpha 1)
        (wait 0.4)

        (spawn! "_spawned"
          (rect :x (spring
                     (+ 90 (* bucket
                              (+ bucket-width
                                 bucket-gap)))
                     :initial 400)
                :y (spring
                     (- (+ bucket-y (/ bucket-height 2))
                        (+ (/ item-height 2)
                           8
                           (* slot
                              (+ item-height 4))))
                     :initial 80)
                :w item-width :h item-height :radius 6
                :fill (palette op-index)
                :scale (tween 0 1)))

        (push! entries
          (:bucket bucket :slot slot))
        (set! op-index (+ op-index 1))

        (wait 0.5)

        (set! arrow-alpha 0)))))
