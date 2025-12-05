; Test 1: Simple value capture
(def get-x (fn [x] (fn [] x)))
(def get-five (get-x 5))
(get-five)

; Test 2: Arithmetic with closure
(def make-adder (fn [x] (fn [y] (+ x y))))
(def add-ten (make-adder 10))
(add-ten 7)

; Test 3: Multiple operations
(def make-multiplier (fn [x] (fn [y] (+ (* x y) x))))
(def times-three-plus (make-multiplier 3))
(times-three-plus 4)
