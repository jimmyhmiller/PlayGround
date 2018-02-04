(require '[experiment-spec.match :refer
           [match make-helper eval-many]])
(require '[experiment-spec.compiler :refer [compile lambda->num]])
(refer-clojure :exclude [eval])


(defn eval [expr]
  (match [expr]
         [n number?] n
         [b boolean?] b
         ('+ x y) (+ (zzevals x) (eval y))))


(eval '(+ 2 (+ 2 2)))
