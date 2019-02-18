(ns account-number.utils
  (:require [clojure.string :as string]
            [account-number.main :as main]
            [account-number.core :as core]
            [clojure.spec.alpha :as spec]
            [clojure.test.check.generators :as gen])
  (:use [account-number.spec]))

(defn underscore [x]
  (if x "_" " "))

(defn pipe [x] 
  (if x "|" " "))

(defn seven-segment->ascii [[a b c d e f g ]]
  (str  " "     (underscore a)  " "     "\n"
       (pipe b) (underscore c) (pipe d) "\n"
       (pipe e) (underscore f) (pipe g)))

(defn account-number->ascii [segments]
  (->> segments
       (map seven-segment->ascii)
       (map #(string/split % #"\n"))
       (apply interleave)
       (partition 9)
       (map string/join)
       (string/join "\n")))

(def valid-account-generator
  (->> (spec/gen :account-number.spec/valid-account-number)
       (gen/fmap #(map core/int->seven-segment %))
       (gen/fmap account-number->ascii)))

(def invalid-account-generator
  (->> (spec/gen :account-number.spec/invalid-account-number)
       (gen/fmap #(map core/int->seven-segment %))
       (gen/fmap account-number->ascii)))

(def valid-or-ill-segment 
  (gen/frequency [[90 (spec/gen :account-number.spec/valid-seven-segment)] 
                  [10 (spec/gen :account-number.spec/ill-formed-seven-segment)]]))

(defn not-valid-account-number-segments [segments]
  (not-every? #(spec/valid? :account-number.spec/valid-seven-segment %) segments))

(def ill-formed-account-generator
  (as-> valid-or-ill-segment g 
    (gen/vector g 9)
    (gen/such-that not-valid-account-number-segments g)
    (gen/fmap account-number->ascii g)))

(def valid-or-invalid-ascii
  (gen/frequency [[50 valid-account-generator] 
                  [50 invalid-account-generator]]))

(defn remove-random-character [ascii]
  (-> ascii
      (string/split #"")
      (update-in [(rand-int (count ascii))]
                 #(if (not= % "\n") " " %))
      string/join))

(def off-by-one-generator 
  (->> valid-or-invalid-ascii
       (gen/fmap remove-random-character)))

(def generate-example-file
  (gen/frequency [[60 valid-account-generator]
                  [10 off-by-one-generator]
                  [20 invalid-account-generator]
                  [10 ill-formed-account-generator]]))


(string/join "\n"
 (gen/sample generate-example-file))




