(ns interpreter.core
  (:require [clojure.core.match :refer [match]])
  (:refer-clojure :exclude [eval]))

(defn eval [expr]
  (match [expr]
         [n :guard number?] n
         [s :guard string?] s
         [b :guard boolean?] b
         [(['+ x y] :seq)] (+ (eval x) (eval y))
         [(['* x y] :seq)] (* (eval x) (eval y))
         [(['- x y] :seq)] (- (eval x) (eval y))
         [(['/ x y] :seq)] (/ (eval x) (eval y))))





(eval '(/ (* (+ 2 5) 2) (- 2 3)))




