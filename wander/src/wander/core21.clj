(ns wander.core21
  (:require [cheshire.core :as json]))


(defn double-it [x]
  (* x 2))

(double-it 2)

(def my-long-list (repeat 1000 :repeat))



(defn my-function-that-throws-an-error-sometimes [x y z]
  (* x (/ x z) (/ 10 (- y (* 2 x)))))



(my-function-that-throws-an-error-sometimes 2 2 3)
(my-function-that-throws-an-error-sometimes 1 2 3)



(json/decode (slurp "http://time.jsontest.com/") true)
