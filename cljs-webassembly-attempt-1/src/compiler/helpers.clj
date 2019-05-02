(ns compiler.helpers)

(defn transform-expr [x]
  (cond
    (seq? x)
    (cond
      (= (first x) 'quote) x
      (= (first x) 'clojure.core/unquote) (second x)
      :else (cons `list (map transform-expr x)))
    :else `(quote ~x)))

(defmacro s-expr [x]
  `~(transform-expr x))

(defmacro s-exprs [& xs]
  `~(transform-expr xs))
