(ns compiler.helpers)

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
