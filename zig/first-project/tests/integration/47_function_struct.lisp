(def Point (: Type)
  (Struct [x Int] [y Int]))

(def make_point (: (-> [Int Int] Point))
  (fn [x y]
    (Point x y)))

(def distance_squared (: (-> [Point] Int))
  (fn [p]
    (let [px (: Int) (. p x)]
      (let [py (: Int) (. p y)]
        (+ (* px px) (* py py))))))

(def p (: Point) (make_point 3 4))
(def result (: Int) (distance_squared p))
(printf (c-str "%lld\n") result)
