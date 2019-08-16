(ns wander.core10
  (:require [meander.match.ir.epsilon :as ir]
            [meander.strategy.epsilon :as r]
            [meander.epsilon :as m]
            [meander.match.epsilon :as match]))


(analyze
 (m/match {:foo 1, :bar 2, :baz 3}
   {:foo 1}
   true))

(analyze 
 (m/match 2
   (m/or {} _)
   :yep))



(m/match []
  (m/or {} _)
  :yep)






(defn analyze-compile
  {:style/indent :defn}
  ([patterns]
   (analyze-compile :find patterns))
  ([kind patterns]
   (analyze-compile kind patterns 'target))
  ([kind patterns expr]
   (let [analyzer (case kind
                    :match match/analyze-match-args
                    :find match/analyze-find-args
                    :search match/analyze-search-args)
         analysis (analyzer (cons 'target patterns))
         matrix (:matrix analysis)
         clauses (:clauses analysis)
         final-clause (:final-clause analysis)
         fail `(fn [] 
                 ~(if (some? final-clause)
                    (ir/compile (match/compile ['target] [final-clause]) nil :match)
                    `(throw (ex-info "non exhaustive pattern match" ))))

         target (gensym "target__")
         ir (match/compile [target] matrix)
         ir* (ir/rewrite 
              (ir/op-bind target (ir/op-eval expr) ir))
         code (ir/compile ir* `(~fail) kind)]
     {:clauses clauses
      :matrix matrix
      :ir ir
      :ir* ir*
      :code code})))

(defmacro analyze [expr]
  (m/match expr
    (~'m/match ?expr & ?body)
    `(analyze-compile :match (quote ~?body) (quote ~?expr))))
