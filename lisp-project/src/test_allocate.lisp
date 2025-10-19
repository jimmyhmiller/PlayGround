(ns test-allocate)

(require [types :as types])

(declare-fn malloc [size I32] -> (Pointer U8))

(def test-fn (: (-> [] I32))
  (fn []
    (printf (c-str "Test 1: make-nil\n"))
    (let [v1 (allocate types/Value (types/make-nil))]
      (printf (c-str "Created nil value\n"))

      (printf (c-str "Test 2: make-cons\n"))
      (let [v2 (allocate types/Value (types/make-symbol (c-str "foo")))
            v3 (types/make-cons v1 v2)]
        (printf (c-str "Created cons\n"))
        0))))

(test-fn)
