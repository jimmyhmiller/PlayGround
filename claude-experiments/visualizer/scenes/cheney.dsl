;; === Cheney's Copying GC ===
;; Semi-space collector with scan/free pointer BFS.
;;
;; Object graph:
;;   root1 -> A -> C -> E
;;   root2 -> B -> D
;;   F is garbage (unreachable)

(scene cheney
  (defc cell-w 70 :min 40 :max 120 :step 1 :category layout)
  (defc cell-h 50 :min 30 :max 80 :step 1 :category layout)
  (defc gap 8 :min 0 :max 20 :step 1 :category layout)
  (defc from-y 180 :min 80 :max 300 :step 1 :category layout)
  (defc to-y 400 :min 250 :max 550 :step 1 :category layout)
  (defc left-x 140 :min 40 :max 300 :step 1 :category layout)

  ;; State machine
  (def step 0)

  ;; Forwarding flags (0 = in from-space, 1 = forwarded)
  (def a-fwd 0)
  (def b-fwd 0)
  (def c-fwd 0)
  (def d-fwd 0)
  (def e-fwd 0)

  ;; Pointers
  (def free-ptr 0)
  (def scan-ptr 0)

  ;; -- From-space background cells --
  (each i (range 6)
    (let cx (+ left-x (* i (+ cell-w gap))))
    (rect :x cx :y from-y
          :w cell-w :h cell-h
          :fill (palette 5) :radius 6
          :opacity 0.3))

  ;; -- From-space objects --
  (rect :id "from-A" :x (+ left-x (* 0 (+ cell-w gap))) :y from-y
        :w cell-w :h cell-h :radius 6
        :fill (palette 0)
        :opacity (spring (if (= a-fwd 1) 0.25 1)))

  (rect :id "from-B" :x (+ left-x (* 1 (+ cell-w gap))) :y from-y
        :w cell-w :h cell-h :radius 6
        :fill (palette 1)
        :opacity (spring (if (= b-fwd 1) 0.25 1)))

  (rect :id "from-C" :x (+ left-x (* 2 (+ cell-w gap))) :y from-y
        :w cell-w :h cell-h :radius 6
        :fill (palette 2)
        :opacity (spring (if (= c-fwd 1) 0.25 1)))

  (rect :id "from-D" :x (+ left-x (* 3 (+ cell-w gap))) :y from-y
        :w cell-w :h cell-h :radius 6
        :fill (palette 3)
        :opacity (spring (if (= d-fwd 1) 0.25 1)))

  (rect :id "from-E" :x (+ left-x (* 4 (+ cell-w gap))) :y from-y
        :w cell-w :h cell-h :radius 6
        :fill (palette 4)
        :opacity (spring (if (= e-fwd 1) 0.25 1)))

  ;; F at slot 5 (garbage — never copied)
  (rect :id "from-F" :x (+ left-x (* 5 (+ cell-w gap))) :y from-y
        :w cell-w :h cell-h :radius 6
        :fill (palette 7))

  ;; -- Reference arrows in from-space (curved arcs above the cells) --
  ;; A -> C (skip 1 slot, medium arc)
  (arrow :from "from-A" :to "from-C"
         :color (palette 0) :stroke-width 1.5 :head-size 7
         :curve -40
         :opacity (spring (if (= a-fwd 1) 0.1 0.7)))
  ;; B -> D (skip 1 slot, medium arc)
  (arrow :from "from-B" :to "from-D"
         :color (palette 1) :stroke-width 1.5 :head-size 7
         :curve -40
         :opacity (spring (if (= b-fwd 1) 0.1 0.7)))
  ;; C -> E (skip 1 slot, medium arc)
  (arrow :from "from-C" :to "from-E"
         :color (palette 2) :stroke-width 1.5 :head-size 7
         :curve -40
         :opacity (spring (if (= c-fwd 1) 0.1 0.7)))

  ;; -- To-space background cells --
  (each i (range 6)
    (let cx (+ left-x (* i (+ cell-w gap))))
    (rect :x cx :y to-y
          :w cell-w :h cell-h
          :fill (palette 6) :radius 6
          :opacity 0.3))

  ;; -- Root pointers --
  (arrow :id "root1"
         :x1 60 :y1 (- from-y 50)
         :x2 (+ left-x (* 0 (+ cell-w gap)))
         :y2 (- from-y 28)
         :color (palette 0) :stroke-width 2.5 :head-size 10)

  (arrow :id "root2"
         :x1 60 :y1 (- from-y 20)
         :x2 (+ left-x (* 1 (+ cell-w gap)))
         :y2 (- from-y 28)
         :color (palette 1) :stroke-width 2.5 :head-size 10)

  ;; -- Scan pointer (triangle below to-space) --
  (triangle :id "scan-tri"
            :x (spring (+ left-x (* scan-ptr (+ cell-w gap))))
            :y (+ to-y 45)
            :size 8
            :fill (palette 2)
            :opacity (spring (if (> step 0) 1 0)))

  ;; -- Free pointer (triangle below to-space, offset right) --
  (triangle :id "free-tri"
            :x (spring (+ left-x (* free-ptr (+ cell-w gap))))
            :y (+ to-y 60)
            :size 8
            :fill (palette 4)
            :opacity (spring (if (> step 0) 1 0)))

  ;; -- Spawned nodes go here --
  (group :id "_spawned")

  ;; -- Click handler --
  ;; Step 1: Copy root A to to-space slot 0
  ;; Step 2: Copy root B to to-space slot 1
  ;; Step 3: Scan A-copy -> copy C to slot 2
  ;; Step 4: Scan B-copy -> copy D to slot 3
  ;; Step 5: Scan C-copy -> copy E to slot 4
  ;; Step 6: Scan D-copy -> no refs, advance scan
  ;; Step 7: Scan E-copy -> no refs, scan==free, done!

  (on :click
    (when (= step 0)
      (sequence
        (set! a-fwd 1)
        (spawn! "_spawned"
          (rect :id "to-A"
                :x (spring (+ left-x (* 0 (+ cell-w gap)))
                           :initial (+ left-x (* 0 (+ cell-w gap))))
                :y (spring to-y :initial from-y)
                :w cell-w :h cell-h :radius 6
                :fill (palette 0)
                :scale (tween 0 1)))
        (set! free-ptr 1)
        (set! step 1)))

    (when (= step 1)
      (sequence
        (set! b-fwd 1)
        (spawn! "_spawned"
          (rect :id "to-B"
                :x (spring (+ left-x (* 1 (+ cell-w gap)))
                           :initial (+ left-x (* 1 (+ cell-w gap))))
                :y (spring to-y :initial from-y)
                :w cell-w :h cell-h :radius 6
                :fill (palette 1)
                :scale (tween 0 1)))
        (set! free-ptr 2)
        (set! step 2)))

    (when (= step 2)
      (sequence
        (set! scan-ptr 1)
        (set! c-fwd 1)
        (spawn! "_spawned"
          (rect :id "to-C"
                :x (spring (+ left-x (* 2 (+ cell-w gap)))
                           :initial (+ left-x (* 2 (+ cell-w gap))))
                :y (spring to-y :initial from-y)
                :w cell-w :h cell-h :radius 6
                :fill (palette 2)
                :scale (tween 0 1)))
        (spawn! "_spawned"
          (arrow :from "to-A" :to "to-C"
                 :color (palette 0) :stroke-width 1.5 :head-size 7
                 :curve 40
                 :opacity (tween 0 0.8)))
        (set! free-ptr 3)
        (set! step 3)))

    (when (= step 3)
      (sequence
        (set! scan-ptr 2)
        (set! d-fwd 1)
        (spawn! "_spawned"
          (rect :id "to-D"
                :x (spring (+ left-x (* 3 (+ cell-w gap)))
                           :initial (+ left-x (* 3 (+ cell-w gap))))
                :y (spring to-y :initial from-y)
                :w cell-w :h cell-h :radius 6
                :fill (palette 3)
                :scale (tween 0 1)))
        (spawn! "_spawned"
          (arrow :from "to-B" :to "to-D"
                 :color (palette 1) :stroke-width 1.5 :head-size 7
                 :curve 40
                 :opacity (tween 0 0.8)))
        (set! free-ptr 4)
        (set! step 4)))

    (when (= step 4)
      (sequence
        (set! scan-ptr 3)
        (set! e-fwd 1)
        (spawn! "_spawned"
          (rect :id "to-E"
                :x (spring (+ left-x (* 4 (+ cell-w gap)))
                           :initial (+ left-x (* 4 (+ cell-w gap))))
                :y (spring to-y :initial from-y)
                :w cell-w :h cell-h :radius 6
                :fill (palette 4)
                :scale (tween 0 1)))
        (spawn! "_spawned"
          (arrow :from "to-C" :to "to-E"
                 :color (palette 2) :stroke-width 1.5 :head-size 7
                 :curve 40
                 :opacity (tween 0 0.8)))
        (set! free-ptr 5)
        (set! step 5)))

    (when (= step 5)
      (sequence
        (set! scan-ptr 4)
        (set! step 6)))

    (when (= step 6)
      (sequence
        (set! scan-ptr 5)
        (set! step 7)))))
