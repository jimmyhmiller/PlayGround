(ns test-simple2)

(require [types :as types])

(def test-fn (: (-> [] I32))
  (fn []
    (let [v (allocate Value (types/make-nil))]
      0)))

;; (test-fn)
