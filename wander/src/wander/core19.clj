(ns wander.core19
  (:require [meander.epsilon :as m]))

(def examples 
  [[:check {:val true :type :boolean :ctx {}}]
   [:check {:val true :type :int :ctx {}}]
   [:synth {:val true :ctx {}}]
   [:synth {:val 1 :ctx {}}]
   [:synth {:val 'x :ctx {'x :int}}]])



(defn bidirectional [expr]
  (m/rewrite expr

    [:check {:val (m/or true false) :type :boolean :ctx ?ctx}] ?ctx

    [:synth {:val (m/or true false)}] :boolean
    [:synth {:val (m/pred int?)}] :int

    [:synth {:val (m/pred symbol? ?x) :ctx {?x ?t}}] ?t))

(map bidirectional examples)

