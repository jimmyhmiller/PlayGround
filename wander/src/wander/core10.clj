(ns wander.core10
  (:require [meander.match.ir.epsilon :as ir]
            [meander.strategy.epsilon :as r]
             [meander.strategy.epsilon :as strat]
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


(analyze
 (m/match '(a a a a)
   (?a ..1)
   !a))



(m/match {{:thing :stuff} :stuff
          :thing ?stuff}

  {:thing ?stuff
   {:thing ?stuff} _}
  [?stuff ?thing])



(m/match {[1 2 3] :hello
          :answer 3}
  {:answer ?x
   [1 2 ?x] ?thing}
  [?thing])

()

(m/search {1 3
           2 3
           3 9}
  {!xs ?x}
  [!xs ?x])


(m/match {:me :thing :stuff {:thing {:a {:b {:c {:d {:e :me}}}}}}}
  {?thing :thing
   :stuff {:thing {:a {:b {:c {:d {:e ?thing}}}}}}}

  [?thing])

(defn favorite-food-info [user foods-by-name]
  (m/match {:user user
            :foods-by-name foods-by-name}
    {:foods-by-name {?food {:popularity ?popularity
                            :calories ?calories}}

     :user
     {:name ?name
      :favorite-food {:name ?food}}}
    {:name ?name
     :favorite {:food ?food
                :popularity ?popularity
                :calories ?calories}}))
(println "\n\n\n\n")

(favorite-food-info 
 {:name "jimmy" :favorite-food {:name :nachos}}
 {:nachos {:popularity :very
           :calories 2000}})


(m/search [1 2 [ 3 4 1]]
  (m/$ [_ ... 1])
  :yep)





(m/search '[(+ 2 3) (* 3 5 0) (- 2 3) (/ 2 0)]
  [_ ... . (m/and (m/scan 0) !steps) . !steps ... . (/ _ 0 :as !steps)]
  !steps)


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