(ns clojurescript-talk.prod
  (:require [clojurescript-talk.core :as core]))

;;ignore println statements in prod
(set! *print-fn* (fn [& _]))

(core/init!)
