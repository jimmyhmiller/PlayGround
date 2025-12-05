(def make-adder (fn [x] (fn [y] (+ x y))))
(def add-five (make-adder 5))
(add-five 3)
