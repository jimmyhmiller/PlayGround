(ns test.enums)
(require [math.utils :as mu])

;; Use enum type from required namespace in a type annotation
;; and use a function that returns an integer instead
(def result (: Int) (mu/get-red-value))
(printf (c-str "%lld\n") result)
