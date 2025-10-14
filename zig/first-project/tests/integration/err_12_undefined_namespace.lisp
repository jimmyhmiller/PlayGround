(ns test.error)
(require [nonexistent.namespace :as nn])

(def x (: Int) nn/value)
