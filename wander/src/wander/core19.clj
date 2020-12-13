(ns wander.core19
  (:require [meander.epsilon :as m]))


(do
  (def examples 
    [[:check {:val true :type :boolean :ctx {}}]
     [:check {:val true :type :int :ctx {}}]
     [:check {:val 1 :type :int :ctx {}}]
     [:check {:val 1 :type :boolean :ctx {}}]
     [:check {:val 'x :type :boolean :ctx {'x :boolean}}]
     [:check {:val 'x :type :boolean :ctx {'x :int}}]
     [:synth {:val true :ctx {}}]
     [:synth {:val 1 :ctx {}}]
     [:synth {:val 'x :ctx {'x :int}}]])



  (defn bidirectional [expr]
    (m/rewrite expr

      [:check {:val (m/or true false) :type :boolean :ctx ?ctx}] ?ctx
      [:check {:val (m/pred int?) :type :int :ctx ?ctx}] ?ctx
      
      [:check {:val (m/pred symbol? ?x) :type ?type :ctx {?x ?type :as ?ctx}}] ?ctx

      [:synth {:val (m/or true false)}] :boolean
      [:synth {:val (m/pred int?)}] :int

      [:synth {:val (m/pred symbol? ?x) :ctx {?x ?t}}] ?t))
  
  (map bidirectional examples))

;; need to handle lambda and apply
