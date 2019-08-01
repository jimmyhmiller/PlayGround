(ns wander.core7
  (:require [meander.epsilon :as m]
            [meander.strategy.epsilon :as r]
            [meander.syntax.epsilon :as r.syntax]
            [meander.match.syntax.epsilon :as match.syntax]))



(def rule-cache (atom {}))
(def execution (atom {:main nil}))
(def rules (atom {}))


(defn reset-execution []
  (reset! execution {}))


(defn set-scope [scope value]
  (when (not= scope :void)
    (swap! execution assoc scope value)))



(defn submit 
  ([in-scope] (submit in-scope (:result (@execution in-scope))))
  ([out-scope data]
   (set-scope out-scope
              {:type :submit 
               :data data})))


(defn match
  ([in-scope] (match in-scope in-scope))
  ([in-scope rule-scope]
   (let [{:keys [type data] :as context} (@execution in-scope)
         rules (@rules rule-scope)
         matched (first
                  (filter (fn [{:keys [id meta] :as rule}]
                            (let [match (:match (@rule-cache id))]
                              
                              (cond
                                meta
                                (match context)
                                
                                (= type :submit)
                                (match data)
                                
                                :else false)))
                          rules))]
     (when matched
       (set-scope rule-scope
                  {:type :match
                   :rule matched
                   :data (if (:meta matched) context data)})))))


(defn rewrite
  ([in-scope]
   (let [{:keys [rule data]} (@execution in-scope)
         {:keys [id out-scope]} rule
         rewrite (:rewrite (@rule-cache id))
         out-scope (or out-scope in-scope)
         result (rewrite data)]
     (set-scope out-scope
                {:type :rewrite
                 :rule rule
                 :data data
                 :result result}))))



(defn run-scope-step 
  ([scope] (run-scope-step scope scope))
  ([scope rule-scope] (run-scope-step scope rule-scope (fn [])))
  ([scope rule-scope cb]
   (cb)
   (when (match scope rule-scope)
     (cb)
     (rewrite rule-scope)
     (cb)
     (submit rule-scope)
     true)))



(defn strict-eval [s]
  (r/until =
    (fn rec [t]
      ((r/pipe
         (r/attempt
          (r/rewrite
           (?f) (~(rec ?f))
           (?f ?x) (~(rec ?f) ~(rec ?x))
           (?f ?x ?y) (~(rec ?f) ~(rec ?x) ~(rec ?y))))
         (r/attempt s))
       t))))

(defn run-strategy [t]
  (submit :main t)
  (run-scope-step :main :main (partial run-scope-step :main :execution))
  (:data (@execution :main)))



(defn run-all [data]
  (reset-execution)
  ((strict-eval run-strategy) 
   data))



(defn compile-rule [{:keys [lhs rhs] :as rule}]
  (let [match (eval `(r/rewrite ~lhs true ~'_ false))
        rewrite (eval `(r/rewrite ~lhs ~rhs))]
    (let [id (gensym "rule")]
      (swap! rule-cache assoc id {:match match :rewrite rewrite})
      (assoc rule 
             :id id))))

(defmacro set-rules
  {:style/indent :defn}
  [scope & current-rules]
  (if (and (keyword? scope) (empty? current-rules))
    `(do (swap! rules assoc ~scope []) nil)
    (let [current-rules (if (map? scope) (conj current-rules scope) current-rules)
          current-rules (map compile-rule current-rules)
          scope (if (keyword? scope) scope :main)]
      `(do (swap! rules 
                  assoc
                  ~scope
                  ~`(quote ~current-rules))
           nil))))


((eval `(r/rewrite :a true ~'_ false))
 :b)


;; Need the ability to match multiple scopes so I can do the time travel
;; Need to allow matching multiple rules per scope
;; Need rules to be a scope


(set-rules
 {:lhs :a :rhs :b}
 {:lhs :b :rhs :c}
 {:lhs :c :rhs :d})


(set-rules
  {:lhs (+ ?x 0) :rhs ?x}
  {:lhs (+ 0 ?x) :rhs ?x}
  {:lhs (- ?x 0) :rhs ?x}
  {:lhs (* 0 ?x) :rhs 0}
  {:lhs (* ?x 0) :rhs 0}
  {:lhs (/ ?x 0) :rhs (error)}
  {:lhs (/ 0 ?x) :rhs 0}
  {:lhs (+ ?x ?y) :rhs ~(+ ?x ?y)}
  {:lhs (- ?x ?y) :rhs ~(- ?x ?y)}
  {:lhs (* ?x ?y) :rhs ~(* ?x ?y)}
  {:lhs (/ ?x ?y) :rhs ~(/ ?x ?y)})





(set-rules
  {:lhs (extract-foos [{:foo (m/or nil !foos)} ...])
   :rhs [{:bar {:foo !foos}} ...]}
  
  {:lhs [{:bar {:foo (m/or 3 !foo-bars)}} ...]
   :rhs {:bar {:foos [!foo-bars ...]}}}
  
  {:lhs {:bar {:foos [!foos]}}
   :rhs {:foo !foos}}

  {:lhs {:bar {:foos [!foos ..2]}}
   :rhs {:foo1 !foos
         :foo2 !foos}}

  {:lhs {:bar {:foos []}}
   :rhs :none}
  

  {:lhs ({:foo ?foo} <> {:bar ?bar})
   :rhs {:foo ?foo
         :bar ?bar}}


  {:lhs (?x + ?y)
   :rhs ~(+ ?x ?y)})


(do
  (println "\n\n")
  (run-all '(extract-foos [{:foo 3} {:foo 6} {:foo 4} {:bar {:foos [5 3 2 5 6]}}])))


(do
  (println "\n\n")
  (run-all '({:foo 1} <> {:bar 3})))


(do
  (println "\n\n")
  (run-all '(2 + 3)))



(println "test" "\n" "asdsa")

(set-rules :execution
  {:meta true
   :lhs ?x
   :rhs ~(prn ?x)})


(set-rules :execution
  {:meta true
   :lhs  {:type :rewrite
          :data ?data
          :result ?result}
   :rhs  ~(do (println ?data) 
              (println ?result)
              (println ""))})

(set-rules :execution
  {:lhs {:type :rewrite
         :data (m/pred seq? ?data)
         :result ?result}
   :rhs ~(do (prn :data ?data "\n" :result ?result "\n\n") 
             (read-line))
   :out-scope :void})


(set-rules :execution
  {:meta true
   :lhs {:type :match
         :rule ?rule}
   :rhs ~(prn ?rule)})

(do
  (println "\n\n")
  (run-all '(/ 3 (- 18 (+ 9 (* 3 (- (+ 3 1) (- (+ 1 3) 3))))))))






;; Scoped rules would be really cool









;; rule:
;;   match:
;;     :a => :b
;;     :b => :c
;;     :c => :d
;; 
;;     
;; rule:
;;   match @execution #meta:
;;     ?current-execution
;; 
;;   match @execution-history
;;    ?history => push(?current-execution ?history)
;; 
;; 
;; rule:
;;   match @debug true
;; 
;;   match @execution #meta:
;;     ?x => print(?x)
;;   
;;   match @console ?cmd // Does this make sense?
;;   
;;   match @execution-history ?history
;; 
;;   match ?cmd ?history:
;;     :left push(?head ?history) do:
;;        @execution-history ?history
;;        @executon ?head
;; 
;;     (:right | :enter) _ => :continue
;; 
;;     :q => @debug false
