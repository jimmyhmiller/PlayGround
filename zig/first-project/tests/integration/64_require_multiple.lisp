(ns test.multiple)
(require [math.utils :as mu])
(require [math.core :as mc])

;; Use definitions from first required namespace
(def x (: Int) mu/value)

;; Use definitions from second required namespace (which itself requires math.utils)
(def y (: Int) mc/double-value)

;; Combine values from both namespaces
(def result (: Int) (+ x y))
(printf (c-str "%lld\n") result)
