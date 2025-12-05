(def get-x (fn [x] (fn [] x)))
(def get-five (get-x 5))
(get-five)
