(ns test.chained)
(require [math.core :as mc])

;; math.core requires math.utils, so this tests chained requires
;; We can use definitions from math.core that themselves use math.utils
(def result1 (: Int) mc/double-value)

;; Use function from math.core that uses math.utils functions
(def result2 (: Int) (mc/add-two 10))

;; Final result
(def result (: Int) (+ result1 result2))