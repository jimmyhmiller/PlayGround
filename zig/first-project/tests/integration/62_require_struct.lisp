(ns test.structs)
(require [math.utils :as mu])

;; Use struct type from required namespace
(def p1 (: mu/Point) (mu/Point 10 20))

;; Use function that returns struct from required namespace
(def p2 (: mu/Point) (mu/make-point 5 15))

;; Access struct fields
(def result (: Int) (+ (. p1 x) (. p2 y)))
(printf (c-str "%lld\n") result)
