(ns alpha-spec.core
  (:require [clojure.spec.alpha :as s]
            [clojure.set :refer [subset?]]))


(def valid-chars (set "0123456789ABCDEFGHJKLMNPRSTUVWXYZ"))

(s/def ::vin (s/and 
              string?
              #(= (count %) 17)
              #(subset? (set %) valid-chars)))


(s/explain ::vin "1234512345O612342")



