(ns index-loader
  (:require [cljs.loader :as loader]
            [index]))

;; This file will be generated at compile time in the future.

(defn ^:export main []
    (index/main))

(loader/set-loaded! :index)
