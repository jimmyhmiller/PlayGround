;; Example 10: Enums and Pattern Matching (Manual)
;; Demonstrates: Enum types, enum variants, conditional logic based on enums

(include-header "stdio.h")
(declare-fn printf [fmt (Pointer U8)] -> I32)

;; Define color enum
(def Color (: Type)
  (Enum Red Green Blue Yellow))

;; Define direction enum
(def Direction (: Type)
  (Enum North South East West))

;; Coord struct for direction vectors
(def Coord (: Type)
  (Struct [x I32] [y I32]))

;; Convert color to string (manual pattern matching with if)
(def color-to-string (: (-> [Color] (Pointer U8)))
  (fn [c]
    (if (= c Color/Red)
        (c-str "Red")
        (if (= c Color/Green)
            (c-str "Green")
            (if (= c Color/Blue)
                (c-str "Blue")
                (c-str "Yellow"))))))

;; Check if color is primary
(def is-primary-color (: (-> [Color] Bool))
  (fn [c]
    (or (= c Color/Red)
        (or (= c Color/Green)
            (= c Color/Blue)))))

;; Get direction vector
(def direction-to-coords (: (-> [Direction] Coord))
  (fn [dir]
    (if (= dir Direction/North)
        (Coord 0 1)
        (if (= dir Direction/South)
            (Coord 0 -1)
            (if (= dir Direction/East)
                (Coord 1 0)
                (Coord -1 0))))))

;; Opposite direction
(def opposite-direction (: (-> [Direction] Direction))
  (fn [dir]
    (if (= dir Direction/North)
        Direction/South
        (if (= dir Direction/South)
            Direction/North
            (if (= dir Direction/East)
                Direction/West
                Direction/East)))))

(def main-fn (: (-> [] I32))
  (fn []
    ;; Test colors
    (printf (c-str "Colors:\n"))
    (printf (c-str "  Red: %s (primary: %d)\n")
            (color-to-string Color/Red)
            (is-primary-color Color/Red))
    (printf (c-str "  Yellow: %s (primary: %d)\n")
            (color-to-string Color/Yellow)
            (is-primary-color Color/Yellow))

    ;; Test directions
    (printf (c-str "\nDirections:\n"))
    (let [north-coords (: Coord) (direction-to-coords Direction/North)]
      (printf (c-str "  North coords: (%d, %d)\n")
              (. north-coords x)
              (. north-coords y)))

    (let [east-coords (: Coord) (direction-to-coords Direction/East)]
      (printf (c-str "  East coords: (%d, %d)\n")
              (. east-coords x)
              (. east-coords y)))

    ;; Test opposite directions
    (printf (c-str "\nOpposite of North is %d (South=%d)\n")
            (opposite-direction Direction/North)
            Direction/South)

    0))

(main-fn)
