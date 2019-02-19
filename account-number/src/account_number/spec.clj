(ns account-number.spec
  (:require [clojure.spec.alpha :as s]
            [clojure.spec.test.alpha :as st]
            [account-number.core :as core]
            [clojure.string :as string]
            [clojure.test.check.generators :as gen]))

(s/def ::digit (s/int-in 0 10))
(s/def ::segment-value #{"|" "_" " "})

(s/def ::seven-segment (s/coll-of boolean? :kind vector? :count 7))
(s/def ::account-number (s/coll-of ::digit :kind vector? :count 9))
(s/def ::ascii-row (s/coll-of (s/coll-of ::segment-value) :count 3))

(s/def ::valid-account-number 
  (s/and ::account-number core/valid-account-number?))

(s/def ::invalid-account-number 
  (s/and ::account-number (complement core/valid-account-number?)))

(s/def ::possible-account-number
  (s/coll-of (s/nilable ::digit) :count 9))

(s/def ::ill-formed-seven-segment 
  (s/and ::seven-segment #(nil? (core/seven-segment->int %))))

(s/def ::valid-seven-segment
  (set (map core/int->seven-segment (range 10))))

(s/fdef core/get-ascii-digit
        :args (s/cat :rows ::ascii-row
                     :x (s/int-in 0 11))
        :ret (s/coll-of string? :count 9))

(s/fdef core/to-seven-segment 
        :args (s/cat :ascii-digit (s/coll-of ::segment-value :count 9))
        :ret ::seven-segment)

(s/fdef core/remove-unnecessary-segments
        :args (s/cat :coll (s/coll-of ::segment-value :count 9))
        :ret (s/coll-of ::segment-value :count 7))

(s/fdef core/rows->seven-segment
        :args (s/cat :rows ::ascii-row)
        :ret (s/coll-of ::seven-segment))

(s/fdef core/int->seven-segment
        :args (s/cat :num ::digit)
        :ret ::seven-segment)

(s/fdef core/seven-segment->int
        :args (s/cat :seven-segment ::seven-segment)
        :ret (s/nilable ::digit))

(s/fdef core/seven-segment->account-number
        :args (s/cat :coll (s/coll-of ::seven-segment))
        :ret (s/coll-of ::possible-account-number))

(s/fdef core/check-sum
        :args (s/cat :numbers ::account-number)
        :ret int?)

(s/fdef core/has-invalid-digits?
        :args (s/cat :numbers ::possible-account-number)
        :ret boolean?)

(s/fdef core/valid-account-number? 
        :args (s/cat :numbers ::possible-account-number)
        :ret boolean?)

(s/fdef core/determine-error-message
        :args (s/cat :numbers ::possible-account-number)
        :ret string?)

(s/fdef core/format-output
        :args (s/cat :numbers ::possible-account-number)
        :ret string?)

(comment
  (st/instrument))


(s/exercise-fn `core/format-output)
