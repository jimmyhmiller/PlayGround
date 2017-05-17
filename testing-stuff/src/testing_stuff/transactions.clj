(ns testing-stuff.transactions
  (:require [clojure.edn :as edn]
            [clj-time.core :as t]
            [clj-time.format :as f]
            [clj-time.predicates :as pr]))


(f/show-formatters)

(defn fix-date [date]
  (clojure.string/replace date " " "T"))

(defn parse-date [date]
  (f/parse (f/formatters :date-hour-minute-second-fraction) date))

(defn weekday? [transaction]
  (let [time (-> transaction :times :when_recorded_local)]
    (pr/weekday? time)))

(defn lift-date [f] 
  (fn [transaction]
    (-> transaction :times :when_recorded_local f)))

(def weekday-transactions
  (->> "/Users/jimmymiller/Desktop/transactions.edn"
       slurp
       edn/read-string
       :transactions
       (map #(update-in % [:times :when_recorded_local] (comp parse-date fix-date)))
       (filter weekday?)
       (filter #(t/after? (-> % :times :when_recorded_local) (t/date-time 2017 3 14)))))

(defn lunch-time? [date]
  (< 11 (t/hour date) 14))

(defn restaurant? [transaction]
  (some #(= (:name %) "Restaurants") (:categories transaction)))

(defn coffee? [transaction]
  (#{"Sp Vardagen Com" "Soho Cafe & Gallery"} (:description transaction)))

(coffee? (nth weekday-transactions 0))


(def lunch-transactions
  (->> weekday-transactions
       (filter (lift-date lunch-time?))
       (filter restaurant?)))

(def coffee-transactions
  (->> weekday-transactions
       (filter coffee?)))

