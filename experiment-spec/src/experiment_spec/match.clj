(ns experiment-spec.match
  (:require [clojure.core.match :as core.match]))

(defn convert-match [case]
  (core.match/match [case]
                    [else :guard keyword?] else
                    [[sym pred]] [sym :guard pred]
                    [s-expr :guard seq?] [(list (into [] s-expr) :seq)]
                    [n] n))

(defmacro match [c & cases]
  `(core.match/match ~c ~@(mapcat (fn [[k v]] [(convert-match k) v]) (partition 2 cases))))

(defn make-helper [f]
  (defmacro eval-many [& exprs]
    `(reduce (fn [_# expr#] (~f expr# {})) nil (quote ~exprs))))
