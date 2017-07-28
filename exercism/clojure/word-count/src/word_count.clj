(ns word-count
  (:require [clojure.string :refer [split replace lower-case]]))

(defn word-count [words]
  (-> words
      lower-case
      (replace #"[^0-9a-zA-Z ]" "")
      (split #" +")
      frequencies))
