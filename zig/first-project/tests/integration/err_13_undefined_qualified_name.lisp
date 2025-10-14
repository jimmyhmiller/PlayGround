(ns test.error)
(require [math.utils :as mu])

;; Try to access a name that doesn't exist in math.utils
(def x (: Int) mu/nonexistent-value)
