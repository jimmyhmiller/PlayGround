(ns prime-multiplication.test-utils
  (:require [clojure.test :refer [report]]
            [clojure.spec.test.alpha :as stest]))

(defn get-result [spec-data]
  (->> spec-data
       first
       :clojure.spec.test.check/ret
       :result))

(defn get-result-error [spec-data]
  (->> spec-data
       first
       :clojure.spec.test.check/ret
       :result-data
       :clojure.test.check.properties/error))

(defn check-spec 
  ([sym]
   (check-spec sym {}))
  ([sym opts]
   (let [result (stest/check sym opts)
         result-error (get-result-error result)]
     (if-not result-error
       (report {:type :pass})
       (report {:type :error
                :message result})))))

(defn check-specs 
  ([syms]
   (check-specs syms {}))
  ([syms opts]
   (doall (map #(check-spec % opts) syms))))
