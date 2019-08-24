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
[{:cols 
  [{:tag :vec, :prt {:tag :prt, :left 
                     {:tag :cat, :elements 
                      [{:tag :lvr, :symbol ?name}
                       {:tag :map, :as nil, 
                        :rest-map nil, 
                        :map {{:tag :lvr, :symbol ?name} 
                              {:tag :lvr, :symbol ?prop}}}]}, 
                     :right {:tag :cat, :elements []}}, :as nil}], 
  :rhs {:value {:name ?name}, :op :return}, :env #{}, :refs {}, :ref-specs {}}]


{:cols [{:tag :set, 
         :elements ({:tag :cat, :elements
                     [{:tag :lit, :value :name} 
                      {:tag :lvr, :symbol ?name}]} 
                    {:tag :cat, :elements [{:tag :lvr, :symbol ?name}
                                           {:tag :lvr, :symbol ?thing}]})} 
        {:tag :any, :symbol _}], 
 :rhs {:value {:name ?name}, :op :return}, :env #{}, :refs {}, :ref-specs {}}


[{:cols [{:tag :map, :as nil, :rest-map nil, 
          :map {{:tag :lvr, :symbol ?name} {:tag :lvr, :symbol ?prop}}}], 
  :rhs {:value {:name ?name}, :op :return}, :env #{{:tag :lvr, :symbol ?name}}, :refs {}, :ref-specs {}}]

(analyze
 (m/match ["jimmy" {"jimmy" :name}]
   [?name {?name ?prop}]
   {:name ?name}))

(analyze)
(m/match {:stuff 1
          :name "jimmy"
          "jimmy" :thing}
  {:name ?name
   ?name ?thing}
  {:name ?name
   :thing ?thing})



[{:cols [{:tag :map,
          :as nil, 
          :rest-map nil, 
          :map {{:tag :lit, :value :name} 
                {:tag :lvr, :symbol ?name}, 
                {:tag :lvr, :symbol ?name}
                {:tag :lvr, :symbol ?thing}}}], 
  :rhs {:value {:name ?name}, :op :return}, :env #{}, :refs {}, :ref-specs {}}]


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
