(ns math.core)
(require [math.utils :as mu])

;; Re-export a value using the required namespace
(def double-value (: Int) (* mu/value 2))

;; Function that uses a function from required namespace
(def add-two (: (-> [Int] Int))
  (fn [x] (mu/add-one (mu/add-one x))))
