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


(defn fixed-point [f] 
  (fn [expr]
    (loop [expr expr
           states [expr]]
      (let [expr* (f expr)]
        (if (= expr* expr)
          states
          (recur expr* (conj states expr*)))))))


(defn n-steps [f n] 
  (fn [expr]
    (loop [expr expr
           states [expr]
           m 0]
      (let [expr* (f expr)]
        (if (= m n)
          states
          (recur expr* (conj states expr*) (inc m)))))))


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

















(defn ck [expr]
  (match expr
    (pred value? ?v) ?v
    ((pred value? ?v) nil) ?v
    ((?m ?n) ?k) (list ?m (list :ar ?n ?k))
    ((pred value? ?v) (:fn (fn [?x] ?m) ?k)) (list (substitute ?m ?x ?v) ?k)
    ((pred value? ?v) (:ar ?n ?k)) (list ?n (:fn ?v ?k))))


((fixed-point ck)
 '(((((fn [x] 
              (fn [y] 
                (fn [z] ((x y) z)))) 
            (fn [x1] x1)) 
           (fn [y1] y1)) 
          3) nil))




(defn not-symbol? [x]
  (not (symbol? x)))


(defn named [name rhs]
  (if (instance? clojure.lang.IObj rhs)
    (with-meta rhs {:name name})
    rhs))

(defn cek [expr]
  (match expr
    (pred value? ?v) 
    (named :value ?v)

    (((pred value? ?v) (pred empty? ?e)) nil) 
    (named :empty ?v)

    (((?m ?n) ?e) ?k) 
    (named :apply (list (list ?m ?e) (list :ar (list ?n ?e) ?k)))

    (((pred not-symbol? ?v) ?e) (:fn ((fn [?x] ?m) ?e') ?k)) 
    (named :fn (list (list ?m (assoc ?e' ?x (list ?v ?e))) ?k))
    
    (((pred not-symbol? ?v) ?e) (:ar (?n ?e') ?k))
    (named :ar (list (list ?n ?e') (list :fn (list ?v ?e) ?k)))

    ((?x ?e) ?k)
    (named :lookup (list (get ?e ?x) ?k))))



;; CEK doesn't have the properties I want for searching. Should I make
;; my own? Can I make it work?

(map (juxt identity meta)
     ((fixed-point cek )
      '(((((fn [x]
             (fn [dec]
               (dec x)))
           0)
          (fn [x] x))
         {}) nil)))


((fixed-point cek)
 '((((((fn [x] 
         (fn [y] 
           (fn [z] ((x y) z)))) 
       (fn [x1] x1)) 
      (fn [y1] y1)) 
     3) {}) nil))
