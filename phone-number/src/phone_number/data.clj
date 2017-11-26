(ns phone-number.data
  (:require [clojure.java.io :as io]
            [clojure.data.csv :as csv])
  (:import [com.google.i18n.phonenumbers 
            PhoneNumberUtil
            PhoneNumberUtil$PhoneNumberFormat]))


(def e164-format PhoneNumberUtil$PhoneNumberFormat/E164)
(def phone-util (PhoneNumberUtil/getInstance))

(def headers [:number :context :name])

; https://github.com/clojure/data.csv#parsing-into-maps
(defn csv-data->maps [csv-data]
  (map zipmap
       (repeat headers)
       csv-data))

(defn read-csv-file [file]
  (with-open [reader (io/reader file)]
    (->> reader
         csv/read-csv
         csv-data->maps
         doall)))

(defn ->e164 [number]
  (try 
    (.format phone-util (.parse phone-util number "US") e164-format)
    (catch Exception e nil)))

(defn valid-e164? [e164]
  (= (count e164) 12))

(defn get-cleaned-data [file]
  (->> file
       (read-csv-file)
       (pmap #(update-in % [:number] ->e164))
       (group-by #(select-keys % [:number :context]))
       (filter #(= (count (second %)) 1))
       (mapcat second)
       (group-by :number)
       (into {})))

(defn valid-number-format? [{:keys [number]}]
  (valid-e164? (->e164 number)))

(defn duplicate-context? [phone-numbers {:keys [number context]}]
  (let [e164 (->e164 number)
        current-entries (get phone-numbers e164)
        contexts (set (map :context current-entries))]
    (contains? contexts context)))

(defn update-if-not-duplicate [phone-numbers {:keys [number] :as entry}]
  (if (not (duplicate-context? phone-numbers entry))
    (update-in phone-numbers [(->e164 number)] conj entry)
    phone-numbers))

(defn find-entry [phone-numbers {:keys [number] :as entry}]
  (->> number
       (get phone-numbers)
       (filter #(= % entry))
       first))
