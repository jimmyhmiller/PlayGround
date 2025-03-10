(ns prime-multiplication.core-test-short
  (:require [clojure.test :refer :all]
            [clojure.spec.test.alpha :as stest]
            [prime-multiplication.core :as core]
            [prime-multiplication.spec :as spec]
            [prime-multiplication.test-utils :as utils]))

(deftest specs
  (utils/check-specs 
   (stest/enumerate-namespace `prime-multiplication.core)
   {:clojure.spec.test.check/opts {:num-tests 50}}))
