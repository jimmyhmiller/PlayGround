(ns prime-multiplication.spec
  (:require [clojure.spec.alpha :as s]
            [prime-multiplication.core :as core]
            [clojure.string :as string]
            [clojure.spec.test.alpha :as stest]
            [orchestra.spec.test :as orch-test]
            [expound.alpha :as expound]))

(s/def ::prime-args
  (s/and (s/cat :n pos-int?  
                :primes (s/coll-of pos-int?))
         #(let [n (:n %)
                primes (:primes %)] 
            (every? (fn [x] (<= x n)) primes))))

(s/fdef core/prime-divisors?
        :args ::prime-args
        :ret boolean?)

(s/fdef core/prime?
        :args ::prime-args
        :ret boolean?)

; This prevents integer overflow on generative tests.
; I could change this to bigints, but generating that many numbers
; would be take way too long.
(def max-number-multiples (int (Math/sqrt Integer/MAX_VALUE)))

(s/fdef core/multiples 
        :args (s/cat :coll (s/coll-of 
                            (s/and pos-int? 
                                   #(< % max-number-multiples))))
        :ret (s/coll-of pos-int?)
        :fn #(let [arg-count (count (-> % :args :coll))
                   ret-count (count (:ret %))]
               (= ret-count (* arg-count arg-count))))

; Without this, our tests would take forever.
; We won't operate on numbers larger than that anyways,
; so no reason to test strings that massive
(def max-format-element-size (count (str Integer/MAX_VALUE)))

(s/fdef core/format-element
        :args (s/and (s/cat :n (s/and pos-int? #(< % max-format-element-size)) 
                            :val pos-int?)
                     #(> (:n %) (count (str (:val %)))))
        :ret (s/and
              string?
              #(= (first %) \space)))

(s/fdef core/calculate-spacing
        :args (s/cat :n (s/and pos-int? #(< % max-number-multiples)))
        :ret pos-int?)

(s/fdef core/format-row
        :args (s/cat :formatter (s/fspec 
                                 :args (s/cat :val pos-int?)
                                 :ret string?)
                     :row (s/coll-of pos-int?))
        :ret string?)

(defn get-row-counts [table]
  (->> (string/split table #"\n")
       (map string/trim)
       (map #(string/split % #" +"))
       (map count)))

(defn first-row-count-matches [n table]
  (let [row-counts (get-row-counts table)]
    (= (first row-counts) n)))

(defn total-count-matches [n table]
  (let [row-counts (get-row-counts table)]
    (= (count row-counts) (inc n))))

(defn all-but-first-count [n table]
  (let [row-counts (get-row-counts table)
        all-but-first-count (set (rest row-counts))]
    (= (first all-but-first-count) (inc n))))

(defn all-but-first-same-count [n table]
  (let [row-counts (get-row-counts table)
        all-but-first-count (set (rest row-counts))]
    (= (count all-but-first-count) 1)))

(s/fdef core/make-table
        :args (s/cat :n (s/and pos-int? #(< % 1000)))
        :ret string?
        :fn (s/and 
             #(first-row-count-matches (-> % :args :n) (:ret %))
             #(total-count-matches (-> % :args :n) (:ret %))
             #(all-but-first-same-count (-> % :args :n) (:ret %))
             #(all-but-first-count (-> % :args :n) (:ret %))))

(comment
  (set! s/*explain-out* expound/printer)
  (orch-test/instrument)
  (core/make-table 10)

  (stest/summarize-results 
   (stest/check (stest/checkable-syms)
                {:clojure.spec.test.check/opts {:num-tests 10}}))

  (stest/check `core/make-table))
