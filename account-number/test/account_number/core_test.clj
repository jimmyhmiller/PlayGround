(ns account-number.core-test
  (:require [clojure.test :refer :all]
            [account-number.main :as main]
            [account-number.core :as core]
            [clojure.java.io :as io]
            [cuerdas.core :refer [<<-]])
  (:import [java.lang AssertionError]))

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
  (is (= scenario-1-answers (main/scenario-1 scenario-1-cases))))

(deftest scenario-3
  (is (= scenario-3-answers (main/scenario-3 scenario-3-cases))))

(deftest split-each-row
  (is (= [] (core/split-each-row [])))
  (is (= [["a" "s" "d" "f"] ["a" "s" "d" "f"]]
         (core/split-each-row ["asdf" "asdf"])))
  (is (= [[" " "_" " " "|"] ["|" " " "_" "|"]]
         (core/split-each-row [" _ |" "| _|"]))))

(deftest split-into-rows
  (is (= [] (core/split-into-rows "")))
  (is (= '([["a" "b" "c"] ["a" "b" "c"] ["a" "b" "c"]]) 
         (core/split-into-rows "abc\nabc\nabc"))))

(def zero-ascii 
  (first 
   (core/split-into-rows
    (<<- " _ 
          | |
          |_|"))))

(def zero-ascii-flattened '(" " "_" " " 
                            "|" " " "|"
                            "|" "_" "|"))

(def eight-zero-ascii 
  (first 
   (core/split-into-rows
    (<<- " _  _ 
          |_|| |
          |_||_|"))))

(deftest get-ascii-digit
  (is (= zero-ascii-flattened (core/get-ascii-digit zero-ascii 0)))
  (is (= zero-ascii-flattened (core/get-ascii-digit eight-zero-ascii 1))))


(deftest hamming-distance
  (is (thrown? AssertionError (core/hamming-distance [1 2] [1])))
  (is (thrown? AssertionError (core/hamming-distance [1 2] [1 2 3])))
  (is (= 0 (core/hamming-distance [] [])))
  (is (= 0 (core/hamming-distance [1] [1])))
  (is (= 1 (core/hamming-distance [1 2] [1 1])))
  (is (= 3 (core/hamming-distance [1 2 2 2] [1 1 1 1]))))


