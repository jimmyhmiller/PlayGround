(ns test.functions)
(require [math.utils :as mu])

;; Use single-parameter function from required namespace
(def result1 (: Int) (mu/add-one 10))

;; Use multi-parameter function from required namespace
(def result2 (: Int) (mu/add 5 7))

;; Combine both
(def result3 (: Int) (mu/add-one (mu/add result1 result2)))
(printf (c-str "%lld\n") result3)
