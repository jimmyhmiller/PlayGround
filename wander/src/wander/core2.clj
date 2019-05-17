(ns wander.core2
  (:require [meander.match.delta :as m]
            [meander.substitute.delta :refer [substitute]]
            [meander.syntax.delta :as syntax]
            [clojure.walk :as walk]
            [meander.strategy.delta :as r]))

(defmacro bup [& body]
  `(r/until = (r/bottom-up (r/trace (r/rewrite ~@body)))))

(defmacro td [& body]
  `(r/until = (r/top-down (r/trace (r/rewrite ~@body)))))



((bup
  (+ 0 . !xs ...)
  (+ . !xs ...))
 '(+ 0 1 2))

(def nn
  (bup
    ('not ('not ?x))
    ?x

    2
    (not (not 3))
    
    ?x ?x))

(def cond-elim
  (bup
   (cond ?pred ?result
         :else ?else)
   (if ?pred ?result ?else)

   (cond ?pred ?result
         . !preds !results ...)
   (if ?pred ?result
       (cond . !preds !results ...))
   
   ?x ?x))



(defmacro bup [& body]
  `(r/until = (r/bottom-up (r/trace (r/rewrite ~@body)))))

(defn repeat-n [n s]
  (apply r/pipe 
         (clojure.core/repeat n s)))

(def unpipe-first
  (repeat-n
   10
   (r/bottom-up 
    (r/rewrite

     (-> ?x (?f . !args ...))
     (?f ?x . !args ...)
     
     (-> ?x ?f)
     (?f ?x)

     (-> ?x ?f . !fs ...)
     (-> (-> ?x ?f) . !fs ...)
     
     ?x ?x))))

(unpipe-first '(-> x f g h))


(def cond-elim
  (r/until =
    (r/bottom-up
     (r/rewrite
      (cond ?pred ?result
            :else ?else)
      (if ?pred ?result ?else)

      (cond ?pred ?result
            . !preds !results ...)
      (if ?pred ?result
          (cond . !preds !results ...))
      
      ?x ?x))))

(cond-elim 
 '(cond true true
        1 1
        3 3
        4 4
        :else false))








((r/rewrite
  (+ 0 . !xs ...)
  (+ . !xs ...))
 '(+ 0 1 2))



(m/match '(+ 1 (+ 1 2))
  (with [%const (pred number? !xs)
         %expr (or (+ %expr %expr) %const)]
        %expr)
  !xs)


(def hiccup 
  [:div
   [:p {"foo" "bar"}
    [:strong "Foo"]
    [:em {"baz" "quux"} "Bar"
     [:u "Baz"]]]
   [:ul
    [:li "Beef"]
    [:li "Lamb"]
    [:li "Pork"]
    [:li "Chicken"]]])

(m/find hiccup
  (with [%h1 [!tags {:as !attrs} . (let !xs %hiccup) ...]
         %h2 (and (let !attrs {}) [!tags . %hiccup ...])
         %h3 !xs
         %hiccup (or %h1 %h2 %h3)]
        %hiccup)
  (substitute [[!tags !attrs !xs] ...])) 
  
