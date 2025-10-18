;; Example 3: Structs and Points
;; Demonstrates: Struct definitions, struct construction, field access, nested structs

(include-header "stdio.h")
(declare-fn printf [fmt (Pointer U8)] -> I32)

;; Define a Point struct
(def Point (: Type)
  (Struct [x I32] [y I32]))

;; Define a Line struct with nested Points
(def Line (: Type)
  (Struct [start Point] [end Point]))

;; Function to calculate distance (Manhattan distance)
(def manhattan-distance (: (-> [Point Point] I32))
  (fn [p1 p2]
    (let [dx (: I32) (- (. p1 x) (. p2 x))]
      (let [dy (: I32) (- (. p1 y) (. p2 y))]
        (+ (if (< dx 0) (- 0 dx) dx)
           (if (< dy 0) (- 0 dy) dy))))))

(def main-fn (: (-> [] I32))
  (fn []
    (let [origin (: Point) (Point 0 0)]
      (let [p1 (: Point) (Point 3 4)]
        (let [line (: Line) (Line origin p1)]
          (printf (c-str "Origin: (%d, %d)\n") (. origin x) (. origin y))
          (printf (c-str "Point 1: (%d, %d)\n") (. p1 x) (. p1 y))
          (printf (c-str "Line start: (%d, %d)\n") (. (. line start) x) (. (. line start) y))
          (printf (c-str "Line end: (%d, %d)\n") (. (. line end) x) (. (. line end) y))
          (printf (c-str "Manhattan distance: %d\n") (manhattan-distance origin p1))
          0)))))

(main-fn)
