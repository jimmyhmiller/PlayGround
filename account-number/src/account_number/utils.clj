(ns account-number.utils
  (:require [clojure.string :as string]
            [account-number.main :as main]
            [account-number.core :as core]
            [clojure.spec.alpha :as s]
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
  (->> (s/gen :account-number.spec/valid-account-number)
       (gen/fmap #(map core/int->seven-segment %))
       (gen/fmap account-number->ascii)))

(def invalid-account-generator
  (->> (s/gen :account-number.spec/invalid-account-number)
       (gen/fmap #(map core/int->seven-segment %))
       (gen/fmap account-number->ascii)))

(def valid-or-ill-segment 
  (gen/frequency [[90 (s/gen :account-number.spec/valid-seven-segment)] 
                  [10 (s/gen :account-number.spec/ill-formed-seven-segment)]]))

(defn not-valid-account-number-segments [segments]
  (not-every? #(s/valid? :account-number.spec/valid-seven-segment %) segments))

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

(defn progress 
  ([n]
   (fn [rf]
     (let [count (atom 0)]
       (fn 
         ([] (rf))
         ([result] (rf result))
         ([result input]
          (swap! count inc)
          (when (zero? (mod @count n))
            (println @count))
          (rf result input))))))
  ([n coll]
   (progress n coll 0))
  ([n coll i]
   (lazy-seq
    (when-let [s (seq coll)]
      (when (zero? (mod i n))
        (println i))
      (cons (first s) (progress n (rest s) (inc i)))))))



(defn digits
  ([n]
   (digits n '()))
  ([n coll]
    (if (<= n 0)
      coll
      (recur (quot n 10) (cons (mod n 10) coll)))))


(defn calc-check [numbers]
  (->> (range 1 9)
       (map * (rest  (reverse numbers)))
       (reduce +)))

(digits 100000000)

(def all-account-numbers
  (time
   (doall 
    (->> (range 300000000 1000000000)
         (take 100001)
         (map digits)
         (progress 1000)
         (filter #(zero? (core/check-sum %)))
         (map calc-check)))))

(comment)
(->> (gen/sample valid-account-generator 100000)
     (string/join "\n")
     core/split-into-rows
     (map core/rows->seven-segment)
     (map (partial into []))
     (map core/generate-possible-account-numbers)
     (filter #(> (count %) 1)))


