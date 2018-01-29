(ns testing-stuff.erorr-handling
  (:require [sc.api]
            [clojure.walk :as walk]))




(def x 2)
(sc.api/letsc 2
              [x y z])


(defn log [x]
  (println x)
  x)

(defn wrap-expr [context expr]
  `(try
     (sc.api/letsc ~context
                   ~expr)
     nil
     (catch Exception e# 
       (do
         {:exeception e#
          :expr (quote ~expr)}))))

(def ^:dynamic *binding-context*)


(defn find-code-with-errors [context expr]
  (->> expr
       (tree-seq seq? identity)
       (map (partial wrap-expr context))
       (map eval)
       (filter identity)
       (map :expr)
       last))


(sc.api/spy {:sc/spy-ep-post-eval-logger 
                    (fn [arg]
                      (println arg))}
                   (+ 2 2))

(defmacro find-error-expr [expr]
  `(let [p# (promise)]
     (try
       (sc.api/spy {:sc/spy-ep-pre-eval-logger identity
                    :sc/spy-ep-post-eval-logger 
                    (fn [{e-id# :sc.ep/id 
                          error# :sc.ep/error
                          value# :sc.ep/value
                          {cs-id# :sc.cs/id 
                           expr# :sc.cs/expr} :sc.ep/code-site}]
                      (deliver p# {:error (find-code-with-errors [e-id# cs-id#] expr#)
                                   :value value#
                                   :cause (when error# (:cause (Throwable->map error#)))}))}
                   ~expr)
       (catch Exception e#))
     (deref p#)))





(defmacro defn-debug [name args & body]
  `(defn ~name ~args
    (find-error-expr ~@body)))







(defn-debug complicated-math [x y z]
  (+ 2 3 (/ x 2) (/ x y (/ 2 z) 
                    (/ z x) (+ x y) (+ 2 (* x y)))))

(complicated-math 1 3 0)
