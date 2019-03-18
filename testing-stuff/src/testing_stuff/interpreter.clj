(ns testing-stuff.interpreter
  (:require [meander.match.gamma :refer [match]]
            [clojure.pprint :as pprint]))


(defn ->state 
  ([expr env]
   (->state expr env nil))
  ([expr env value]
   {:value value
    :expr expr
    :env env}))


(defn ->value [value env]
  {:value value
   :expr nil
   :env env})



(defn set-var [env var val]
  (assoc env var val))

(defn interp
  ([expr] 
   (interp expr {}))
  ([expr env]
   (match expr
     (pred nil? ?x)                     (->value ?x env)
     (pred number? ?x)                  (->value ?x env)
     (pred string? ?x)                  (->value ?x env)
     (pred boolean? ?x)                 (->value ?x env)
     (pred symbol? ?x)                  (->value (get env ?x ?x) env)
     (if ?pred ?t ?f)                   (let [{:keys [value]} (interp ?pred env)]
                                          (cond 
                                            (true? value) (list (->state ?t env))
                                            (false? value) (list (->state ?f env))
                                            :else (list (->state ?t env) 
                                                        (->state ?f env))))
     ('let [(pred symbol? ?var) ?val] 
      ?body)                            (let [{:keys [:value]} (interp ?val)] 
                                          (list (->state ?body
                                                         (set-var env ?var value)))))))



(defn step2
  ([expr]
   (step2 expr {}))
  ([expr env]
   (let [{:keys [expr env]} (first (interp expr env))]
     (interp expr env))))

(defn examples []
  [(interp 2 {})
   (interp "test" {})
   (interp true {})
   (interp 'x {})
   (interp 'x {'x 2})
   (interp '(if true 1 2) {})
   (interp '(if false 1 2) {})
   (interp '(if n 1 2) {})
   (interp '(if n (if p 1 2) (if q 2 3)) {})
   (interp '(if n (if p 1 2) (if q 2 3)) {'n true})
   (interp '(let [x true]
              (if x
                (if false 1 2)
                3)))
   (step2  '(let [x true]
              (if x
                (if false 1 2)
                3)))])



;; Wouldn't it be cool to be able to pattern match on values going
;; through an interpreter? That way you could do data level debugging,
;; which I think might be more interesting and more useful than
;; breakpoint debugging.


(add-watch #'interp :examples
           (fn [_ _ _ _]
             (pprint/pprint (examples))))

(add-watch #'examples :examples
           (fn [_ _ _ _]
             (pprint/pprint (examples))))
