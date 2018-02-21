(ns ^:figwheel-no-load live-code.dev
  (:require
    [live-code.core :as core]
    [devtools.core :as devtools]))

(devtools/install!)

(enable-console-print!)

(core/init!)
