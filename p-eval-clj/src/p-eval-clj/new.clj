(ns p-eval-clj.new
  (:require [clojure.walk :as walk]))



(def special-syms 
  '#{case*
     try
     finally
     loop*
     do
     letfn*
     if
     let*
     fn*
     recur
     set!
     var
     quote
     catch
     throw
     def})



(defn analyze [expr]
  (cond (seq? expr)
        (case (first expr)
          if {:op :if
              :pred (analyze (nth expr 1))
              :then (analyze (nth expr 2))
              :else (analyze (nth expr 3))}
          let* {:op :let
                :args (analyze-bindings (nth expr 1) analyze)
                :children (analyze-many (rest (rest expr)) analyze)}
          def {:op :def
               :name (analyze (nth expr 1))
               :body (analyze (nth expr 2))}
          fn* {:op :fn
               ;; Need to handle the fact that fns can have multiple arity
               :args (analyze-many (nth (nth expr 1) 0) analyze)
               :body (analyze-many (rest (nth expr 1)) analyze) }
          {:op :call
           :fn (analyze (nth expr 0))
           :args (analyze-many (rest expr) analyze)})
        (number? expr) {:op :const
                        :type :number
                        :value expr}
        (string? expr) {:op :const
                        :type :string
                        :value expr}
        (keyword? expr) {:op :const
                         :type :keyword
                         :value expr}
        (symbol? expr) {:op :symbol
                        :value expr}
        (nil? expr) {:op nil}
        (map? expr) {:op :map 
                     :entries (analyze-entries expr analyze)}
        (vector? expr) {:op :vector
                        :entries (analyze-many expr analyze)}))


(defn analyze-many [coll analyze]
  (if (empty? coll)
    coll
    (cons (analyze (first coll)) 
          (analyze-many (rest coll) analyze))))

(defn analyze-entries [coll analyze]
  (if (empty? coll)
    (list)
    (let [entry (first coll)
          key-value (analyze (key entry))
          val-value (analyze (val entry))]
      (cons [key-value val-value] (analyze-entries (rest coll) analyze)))))

(defn analyze-bindings [coll analyze]
  (if (empty? coll)
    (list)
    (let [key-value (analyze (first coll))
          val-value (analyze (second coll))]
      (cons [key-value val-value] (analyze-bindings (rest coll) analyze)))))




(defn macroexpand-special [expr]
  (walk/prewalk (fn [x]
                  (cond
                    ;; case expands to case* which is inscrutable and undocumented
                    (and (seq? x) (= (first x) 'case))
                    (cons 'case (map macroexpand-special (rest x)))
                    (seq? x)
                    (macroexpand x)
                    :else x))
                expr))



(analyze
 (macroexpand-special '(defn analyze [expr]
                         (cond (seq? expr)
                               (case (first expr)
                                 if {:op :if
                                     :pred (analyze (nth expr 1))
                                     :then (analyze (nth expr 2))
                                     :else (analyze (nth expr 3))}
                                 let* {:op :let
                                       :args (analyze-bindings (nth expr 1) analyze)
                                       :children (analyze-many (rest (rest expr)) analyze)}
                                 def {:op :def
                                      :name (analyze (nth expr 1))
                                      :body (analyze (nth expr 2))}
                                 fn* {:op :fn
                                      ;; Need to handle the fact that fns can have multiple arity
                                      :args (analyze-many (nth (nth expr 1) 0) analyze)
                                      :body (analyze-many (rest (nth expr 1)) analyze) }
                                 {:op :call
                                  :fn (analyze (nth expr 0))
                                  :args (analyze-many (rest expr) analyze)})
                               (number? expr) {:op :const
                                               :type :number
                                               :value expr}
                               (string? expr) {:op :const
                                               :type :string
                                               :value expr}
                               (keyword? expr) {:op :const
                                                :type :keyword
                                                :value expr}
                               (symbol? expr) {:op :symbol
                                               :value expr}
                               (map? expr) {:op :map 
                                            :entries (analyze-entries expr analyze)}
                               (vector? expr) {:op :vector
                                               :entries (analyze-many expr analyze)}))))



