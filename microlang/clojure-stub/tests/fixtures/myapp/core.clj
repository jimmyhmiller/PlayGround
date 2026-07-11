(ns myapp.core
  (:require [util.math :as m]
            [util.helper :refer [greet]]))
(defn area [r] (* m/pi (m/square r)))
(defn hello [] (greet "world"))
