(ns scheme-clojure.scratchpad
  (:require [scheme-clojure.core :refer [scheme->clojure]]))



(scheme->clojure
 (define x (cons (list 1 2) (list 3 4)))
 (define (count-leaves x)
   (cond ((null? x) 0)
         ((not (pair? x)) 1)
         (else (+ (count-leaves (car x))
                  (count-leaves (cdr x)))))))
