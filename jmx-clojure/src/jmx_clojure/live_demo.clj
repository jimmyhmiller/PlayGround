(ns jmx-clojure.live-demo
  (:require [clojure.repl :refer [source]]))





(def nums (range 100))

(defn fizz? [n]
  (if (mod n 3) "fizz" n))

(fizz? 3)






