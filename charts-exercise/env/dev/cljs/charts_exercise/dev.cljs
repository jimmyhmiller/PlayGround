(ns ^:figwheel-no-load charts-exercise.dev
  (:require
    [charts-exercise.core :as core]
    [devtools.core :as devtools]))


(enable-console-print!)

(devtools/install!)

(core/init!)
