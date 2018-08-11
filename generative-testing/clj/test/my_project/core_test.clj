(ns my-project.core-test
  (:require [my-project.core :as core]
            [clojure.test.check :as tc]
            [clojure.test.check.generators :as gen]
            [clojure.test.check.properties :as prop]
            [clojure.test.check.clojure-test :refer [defspec]]))

(defspec reverse-reverse
  100
  (prop/for-all [xs (gen/vector gen/int)]
    (= (core/reverse (core/reverse xs)) xs)))
