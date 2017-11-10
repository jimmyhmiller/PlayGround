(ns account-number.core-test
  (:require [clojure.test :refer :all]
            [account-number.main :as main]
            [clojure.java.io :as io]
            [cuerdas.core :refer [<<-]]))


(def scenario-1-cases (slurp (io/resource "scenario-1-test.txt")))

(def scenario-1-answers
  (<<- "000000000
        111111111
        222222222
        333333333
        444444444
        555555555
        666666666
        777777777
        888888888
        999999999
        123456789"))

(def scenario-3-cases (slurp (io/resource "scenario-3-test.txt")))

(def scenario-3-answers
  (<<- "000000051
        49006771? ILL
        1234?678? ILL
        664371495 ERR"))

(deftest scenario-1
  (testing "Properly parse scenario-1"
    (is (= scenario-1-answers (main/scenario-1 scenario-1-cases)))))


(deftest scenario-3
  (testing "Properly parse scenario-3"
    (is (= scenario-3-answers (main/scenario-3 scenario-3-cases)))))
