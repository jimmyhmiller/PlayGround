(ns wander.core3
  (:import (clojure.lang Sequential IPersistentMap))
  (:require [clojure.string :as string]
            [meander.strategy.delta :as r]
            [meander.strategy.epsilon :as re]
            [meander.epsilon :as me]
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
     (println "strict" t)
     ((r/pipe
       (r/attempt
        (r/rewrite

         (if true ?t) ~(rec ?t)
         (if false ?t) nil
         (if true ?t ?f) ~(rec ?t)
         (if false ?t ?f) ~(rec ?f)
         
         (?f) (~(rec ?f))
         (?f ?x) (~(rec ?f) ~(rec ?x))
         (?f ?x ?y) (~(rec ?f) ~(rec ?x) ~(rec ?y))
         (?f ?x ?y ?z) (~(rec ?f) ~(rec ?x) ~(rec ?y) ~(rec ?z))))
       (r/choice
        (r/rewrite
         ((io . !xs ...) ?x) ((io . !xs ...) ~(rec ?x))
         (?f (io . !xs ...)) (~(rec ?f) (io . !xs ...))
         ((io . !xs ...) ?x ?y) ((io . !xs ...) ~(rec ?x) ~(rec ?y))
         (?f (io . !xs ...) ?y) (~(rec ?f) (io . !xs ...) ~(rec ?y))
         (?f ?x (io . !xs ...)) (~(rec ?f) ~(rec ?x) (io . !xs ...)))
        (r/attempt s)))
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





(do
  (println "\n\n\n\n")
  ((strict-eval little-lang)
   '(if (= 1 1)
      (if (= (io get-data) 1)
        (if (= 1 1)
          (io more-data))
        false)
      true)))




(do
  (println "\n\n\n\n")
  ((lazy-eval little-lang)
   '(if (= 1 1)
      (if (= 3 1)
        (loop)
        false)
      true)))



;; Bug. Need to add 

((re/rewrite
  [[:control []] (prog . (me/and (go . _ ...) ..1) !gos)]
  [[:control [!gos ...]]])

 '[[:control []]
   (prog
    (go
      (dispatch :ping)
      (wait :pong))
    (go
      (wait :ping)
      (dispatch :pong)))])






((r/n-times 2
   (re/rewrite
    [[:control []] (prog . (go . !actions ..!n) ..1)]
    [:control . [!actions ..!n] ...]

    [:control . [!candidates & !remaining] ..!n]
    [:choice [!candidates ...]
     [!remaining ..!n]]

   
    ))

 '[[:control []]
   (prog
    (go
      (dispatch :ping)
      (wait :pong))
    (go
      (wait :ping)
      (dispatch :pong)))])
