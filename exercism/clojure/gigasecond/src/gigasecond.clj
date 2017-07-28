(ns gigasecond
  (:require [clj-time.core :as t]))


(defn date->tuple [date]
  [(t/year date) (t/month date) (t/day date)])

(defn from [year month day]
  (-> (t/date-time year month day)
      (t/plus (t/seconds 1000000000))
      date->tuple))
