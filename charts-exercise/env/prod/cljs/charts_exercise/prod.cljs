(ns charts-exercise.prod
  (:require
    [charts-exercise.core :as core]))

;;ignore println statements in prod
(set! *print-fn* (fn [& _]))

(core/init!)
