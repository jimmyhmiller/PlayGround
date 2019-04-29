(ns wander.core2
  (:require [meander.match.gamma :as m]
            [meander.substitute.gamma :refer [substitute]]
            [meander.syntax.gamma :as syntax]
            [clojure.walk :as walk]))


(defn transform-expr [x]
  (cond
    (symbol? x) `(quote ~x)
    (seq? x)
    (cond
      (= (first x) 'quote) x
      (= (first x) 'clojure.core/unquote) (second x)
      :else (cons `list (map transform-expr x)))
    :else x))

(defmacro s-expr [x]
  `~(transform-expr x))


(defn compile-expr [expr]
  (if (number? expr)
    '(i32.const expr)))



(defn compile [expr]
  (s-expr (module 
           (export "main" (func $main))
           (func $main (result i32)
                 ~(compile-expr expr)))))


(compile 13)

