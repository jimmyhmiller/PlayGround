; Test closures with captured values

; Test 1: Simple value capture
(def get-x (fn [x] (fn [] x)))
(def get-five (get-x 5))
(get-five)

; Test 2: Arithmetic with captured value
(def make-adder (fn [x] (fn [y] (+ x y))))
(def add-five (make-adder 5))
(add-five 3)

; Test 3: Complex arithmetic with captured value
(def make-multiplier (fn [x] (fn [y] (+ (* x y) x))))
(def times-three-plus (make-multiplier 3))
(times-three-plus 4)
