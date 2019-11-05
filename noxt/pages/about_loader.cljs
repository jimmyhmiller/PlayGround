(ns about-loader
  (:require [cljs.loader :as loader]
            [about]))

;; This file will be generated at compile time in the future.

(defn ^:export main []
    (about/main))

(loader/set-loaded! :about)

