(ns wander.core3
  (:import (clojure.lang Sequential IPersistentMap))
  (:require [clojure.string :as string]
            [meander.strategy.delta :as r]
            [meander.match.delta :as m]
            [clojure.walk :as walk]))



(m/search [1 [2 3] 3 [4 7]]
  (scan (scan ?x))
  ?x)


(with [(element ??tag) [??tag {:as _} . _ ...]]
   {:tag ?tag
    :hiccup (element ?tag)})



(defn log [& xs]
  (println xs)
  (last xs))




(defn repeat-n 
  {:style/indent :defn}
  [n s]
  (apply r/pipe (repeat n s)))

(defn strict-eval [s]
  (r/until =
   (fn rec [t]
     ((r/pipe
       (r/choice
        (r/rewrite
         (if true ?t ?f) ~(rec ?t)
         (if false ?t ?f) ~(rec ?f)
         (if ?pred ?t ?f) (if ~(rec ?pred) ?t ?f)
         (?f) (~(rec ?f))
         (?f ?x) (~(rec ?f) ~(rec ?x))
         (?f ?x ?y) (~(rec ?f) ~(rec ?x) ~(rec ?y)))
        (r/attempt s))
       (r/attempt s))
      t))))



(defn lazy-app [rec]
  (fn [t]
    (log "lazy" t)
    ((r/rewrite
       (?f) (~(rec ?f))
       (?f ?x) (~(rec ?f) ?x)
       (?f ?x ?y) (~(rec ?f) ?x ?y))
     t)))

(defn lazy-eval [s]
  (r/until =
    (fn rec [t]
      (log "eval" t)
      ((r/pipe
        (r/choice
         s
         (lazy-app rec)
         (r/attempt (r/one (r/pipe (lazy-app rec) (r/attempt s)))))
        (r/attempt s))
       t))))



(defn little-lang [t]
  (log "little" t)
  ((r/rewrite
    (if true ?t) ?t
    (if false ?t) nil 
    (if true ?t ?f) ?t
    (if false ?t ?f) ?f
    (= ?x ?x) true
    (= ?x ?y) false
    (ignore ?x ?y) ?y
    (loop) ~(throw (ex-info "haha" {})))
   t))






((strict-eval little-lang)
 '(ignore (loop) 3))

((lazy-eval little-lang)
 '(if (= 1 2)
    (if (= 3 1)
      (loop)
      false)
    true))

