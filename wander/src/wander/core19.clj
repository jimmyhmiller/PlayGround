(ns wander.core19
  (:require [meander.epsilon :as m]))


(def examples
  [[:check {:val true :type :boolean :ctx {:simple-type true}}]
   [:check {:val true :type :int :ctx {:simple-type false}}]
   [:check {:val 1 :type :int :ctx {:simple-int true}}]
   [:check {:val 1 :type :boolean :ctx {:simple-int false}}]
   [:check {:val 'x :type :boolean :ctx {'x :boolean}}]
   [:check {:val 'x :type :boolean :ctx {'x :int}}]

   [:check {:val '(fn [x] 1) :type [:-> :int :int] :ctx {:fn true}}]
   [:check {:val '(fn [x] true) :type [:-> :int :int] :ctx {:fn false}}]


   [:check {:val '(fn [f] (fn [x] (f x)))
            :type [:-> [:-> :int :int] [:-> :int :int]]
            :ctx {:fn-nested true}}]

   [:check
    {:val '(f x),
     :type :int,
     :ctx '{:fn-nested true, f [:-> :int :int], x :int}}]

   [:synth {:val '(f 2) :ctx {'f [:-> :int :int]}}]
   [:synth {:val '(f false) :ctx {'f [:-> :int :int]}}]

   [:synth {:val true :ctx {}}]
   [:synth {:val 1 :ctx {}}]
   [:synth {:val 'x :ctx {'x :int}}]])



(defn bidirectional [expr]
  (m/rewrite expr

    [:check {:val (m/or true false) :type :boolean :ctx ?ctx}]
    ?ctx

    [:check {:val (m/pred int?) :type :int :ctx ?ctx}]
    ?ctx

    [:check {:val (m/pred symbol? ?x) :type ?type :ctx {?x ?type :as ?ctx}}]
    ?ctx

    (m/and
     [:check {:val (fn [?x] ?body) :type [:-> ?t1 ?t2] :ctx ?ctx}]
     (m/let [(m/cata {:as ?ctx2}) (m/subst [:check {:val ?body :type ?t2 :ctx {?x ?t1 & ?ctx}}])]))
    ?ctx2

    ;; Otherwise, synthesize the type and check that.
    (m/and [:check {:val ?val :ctx ?ctx :type ?t}]
           (m/let [(m/cata ?t) [:synth {:val ?val :ctx ?ctx}]]))
    ?ctx

    [:synth {:val (m/or true false)}]
    :boolean

    [:synth {:val (m/pred int?)}]
    :int

    [:synth {:val (m/pred symbol? ?x) :ctx {?x ?t}}]
    ?t

    (m/and
     [:synth {:val (?f ?x) :ctx ?ctx}]
     (m/let [(m/cata [:-> ?t1 ?t2]) [:synth {:val ?f :ctx ?ctx}]
             (m/cata {}) [:check {:val ?x :type ?t1 :ctx ?ctx}]]))
    ?t2


    ?x [:fail ?x]))

(map bidirectional examples)
