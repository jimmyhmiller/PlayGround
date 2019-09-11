(ns noxt.about
  (:require [cljs.loader :as loader]))

(enable-console-print!)

(defn ^:export main []
  [:h1 "About"])



(loader/set-loaded! :noxt.about)
