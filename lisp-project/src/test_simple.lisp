(ns test-simple)

(require [types :as types])

(def test-fn (: (-> [] I32))
  (fn []
    (let [v (allocate types/Value (types/make-nil))]
      (printf (c-str "Created value\n"))
      0)))

(test-fn)
