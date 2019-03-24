(ns plt-redex.core
  (:require [meander.match.gamma :refer [match]]
            [clojure.walk :as walk]))

(defn value? [x]
  (or (number? x)
      (symbol? x)
      (and (coll? x) (= (first x) 'fn))))


(defn substitute [body var value]
  (walk/postwalk 
   (fn [x] 
     (if (= x var)
       value
       x)) body))



(defn cc [expr]
  (match expr
    (pred value? ?v) ?v
    ((pred value? ?v) nil) ?v
    (((pred (complement value?) ?m) ?n) ?e) (list ?m (cons (list :hole ?n) ?e))
    (((pred value? ?v) (pred (complement value?) ?m)) ?e) (list ?m (cons (list ?v :hole) ?e))
    (((fn [?x] ?body) ?y) ?e) (list (substitute ?body ?x ?y) ?e)
    ((pred value? ?v) ((?u :hole) . !e ...)) (list (list ?u ?v) (seq !e))
    ((pred value? ?v) ((:hole ?n) . !e ...)) (list (list ?v ?n) (seq !e))))






(cc
 (cc
  (cc
   (cc
    (cc
     (cc
      (cc
       (cc
        '(((((fn [x] (fn [y] (fn [z] x))) 1) 2) 3) nil)))))))))



(cc '((((fn [x] (fn [x] x) 1)) 1) nil))

(cc
 (cc '(((fn [x] x) 1) nil)))



(cc)
(cc
 '(((fn [x] x) (fn [x] x)) nil))






(defn scc [expr]
  (match expr
    (pred value? ?v) ?v
    ((pred value? ?v) nil) ?v
    ((?m ?n) ?e) (list ?m (cons (list :hole ?n) ?e))
    ((pred value? ?v) (((fn [?x] ?m) :hole) . !e ...)) (list (substitute ?m ?x ?v) (seq !e))
    ((pred value? ?v) ((:hole ?n) . !e ...)) (list ?n (cons (list ?v :hole) (seq !e)))))




(scc
 (scc
  (scc
   (scc
    '(((fn [x] x) (fn [x] x)) nil)))))

(scc
 (scc
  (scc
   (scc
    (scc
     (scc
      (scc
       (scc
        (scc
         (scc
          (scc
           (scc
            (scc
             (scc
              (scc
               (scc
                '(((((fn [x] 
                       (fn [y] 
                         (fn [z] ((x y) z)))) 
                     (fn [x1] x1)) 
                    (fn [y1] y1)) 
                   3) nil)))))))))))))))))






(ck
 '(((fn [x] x) (fn [x] x)) nil))












(defn ck [expr]
  (match expr
    (pred value? ?v) ?v
    ((pred value? ?v) nil) ?v
    ((?m ?n) ?k) (list ?m (list :ar ?n ?k))
    ((pred value? ?v) (:fn (fn [?x] ?m) ?k)) (list (substitute ?m ?x ?v) ?k)
    ((pred value? ?v) (:ar ?n ?k)) (list ?n (:fn ?v ?k))))

(ck
 (ck
  (ck
   (ck
    (ck
     (ck
      (ck
       '(((((fn [x] 
              (fn [y] 
                (fn [z] ((x y) z)))) 
            (fn [x1] x1)) 
           (fn [y1] y1)) 
          3) nil))))))))
