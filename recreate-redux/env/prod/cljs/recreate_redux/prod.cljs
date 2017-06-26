(ns recreate-redux.prod
  (:require [recreate-redux.core :as core]))

;;ignore println statements in prod
(set! *print-fn* (fn [& _]))

(core/init!)
