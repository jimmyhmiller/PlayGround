(ns scheme-clojure.core
  (:require [clojure.core.match :refer [match]]
            [clojure.walk :refer [postwalk macroexpand-all]]))



(defn lambda->fn [form]
  (if (and (seq? form) (= (first form) 'lambda))
    (let [[_ args body] form]
      `(fn [~@args] (~@body)))
    form))


(defn define->defn [form]
  (if (and (seq? form) (= (first form) 'define) (seq? (second form)))
    (let [[_ [fn-name & args] body] form]
      `(defn ~fn-name [~@args] (~@body)))
    form))


(defn define->def [form]
  (if (and (seq? form) (= (first form) 'define) (not (seq? (second form))))
    (let [[_ var-name body] form]
      `(def ~var-name ~body))
    form))


(defn flatten-cond [[c & cases]]
  (cons 'cond (reduce concat cases)))


(defn replace-cond [form]
    (if (and (seq? form) (= (first form) 'cond))
      (flatten-cond form)
      form))


(defn null? [x]
  (and (seq? x) (empty? x)))


(defn replace-null? [form]
  (if (and (seq? form) (= (first form) 'null?))
    (let [[_ body] form]
      `(and (seq? ~body) (empty? ~body)))
    form))

(defn replace-keywords [word]
  (match [word]
         ['remainder] 'mod
         ['else] :else
         ['car] 'first
         ['cdr] 'rest
         ['pair?] 'seq?
         :else word))

(defn scheme [form]
  (->> form
       (postwalk (comp replace-keywords replace-cond define->def lambda->fn define->defn replace-null?))))


(defmacro scheme->clojure [& form]
  (let [code (reverse (map scheme form))]
  `(do ~@code)))






