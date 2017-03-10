(ns reagent-poc.prod
  (:require [reagent-poc.core :as core]))

;;ignore println statements in prod
(set! *print-fn* (fn [& _]))

(core/init!)
