(ns try-language.macros
  (:require [cemerick.cljs.test :as t]))

(defmacro obj->
  [x & forms]
  (loop [x x, forms forms]
    (if forms
      (let [form (first forms)
            threaded (if (seq? form)
                       (with-meta `(~x ~(first form) ~@(next form)) (meta form))
                       (list x form))]
        (recur threaded (next forms)))
      x)))

(defn dispatcher [obj message & args]
  (cond
   (= message :methods)
   (keys obj)
   (= message :extend)
   (partial dispatcher (merge obj (first args)))
   (t/function? obj)
   (apply obj (cons message args))
   :else
   (apply (obj message)
          (cons (partial dispatcher obj) args))))

(defmacro defclass
  ([fn-name body]
   `(def ~fn-name
      (partial dispatcher ~body)))
  ([fn-name params body]
   `(defn ~fn-name ~params
      (partial dispatcher ~body))))
