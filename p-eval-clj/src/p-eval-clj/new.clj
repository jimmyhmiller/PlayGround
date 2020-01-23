(ns p-eval-clj.new
  (:require [clojure.walk :as walk]))



;; NOTES:
;;
;; I have been doing a silly pass the function as an arg as an attempt
;; to ignore def that is empty. Probably just want to go ahead and
;; deal with that instead.


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
      (cons [key-value val-value] (analyze-bindings (rest (rest coll)) analyze)))))



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



(defn add-var [symbol expr env]
  (cons [symbol expr] env))


(defn lookup [symbol env]
  (cond (empty? env)
    symbol
    (= (first (first env)) symbol)
    (second (first env))
    :else (lookup symbol (rest env))))


(defn resolve-symbol [symbol env]
  ;; make this work for non-built-in
  (or (resolve symbol) (analyze symbol)))

(defn evalable-seq? [coll env evalable?]
  (if (empty? coll)
    true
    (and (evalable? (first coll) env)
         (evalable-seq? (rest coll) env evalable?))))


(defn built-in? [sym]
  (boolean? (resolve sym)))

(defn evalable? [expr env]
  (case (:op expr)
    :const true
    :call (evalable-seq? (:args expr) env evalable?)
    :symbol (or (keyword? (:value expr))
                (let [value (lookup expr env)]
                  (and (not= value expr)
                       (evalable? value env)))
                (built-in? (:value expr)))))


(defn extract-expr [pair]
  (first pair))

(def p-eval)

(defn p-eval-many-ignore-env [coll env]
  (if (empty? coll)
    '()
    (cons (extract-expr (p-eval (first coll) env))
          (p-eval-many-ignore-env (rest coll) env))))

(defn extract-value-many [coll]
  (if (empty? coll)
    '()
    (cons (:value (first coll))
          (extract-value-many (rest coll)))))

(defn p-eval-call [expr env]
  (if (evalable? expr env)
    ;; handle non-built-in
    [(analyze
      (apply (resolve-symbol (:value (extract-expr (p-eval (:fn expr) env))) env)
             (extract-value-many (p-eval-many-ignore-env (:args expr) env)))) 
     env]
    
    ;; specialize function
    [{:op :call
      :fn (extract-expr (p-eval (:fn expr) env))
      :args (extract-value-many (p-eval-many-ignore-env (:args expr) env))}
     env]))



(defn p-eval-let [expr env]
  (cond (empty? (:args expr))
        ;; is ignore right here?
        ;; In what cases would there be multiple? 
        ;; Only for side effects? Do we care? What do we do?
        [(p-eval-many-ignore-env (:children expr) env) env] 
        (empty? (rest (:args expr)))
        (p-eval-let {:op :let
                     :args (list)
                     :children (:children expr)}
                    (add-var (first (first (:args expr)))
                             (extract-expr (p-eval (second (first (:args expr))) env))
                             env))))


(defn collect-def [expr env]
  (case (:op expr)
    :def (add-var (:name expr) (:body expr) env)
    env))


(defn collect-defs [exprs env]
  (if (empty? exprs)
    env
    (collect-defs (rest exprs) (collect-def (first exprs) env))))

(defn p-eval 
  ([expr]
   (p-eval expr '()))
  ([expr env]
   (case (:op expr)
     :const [expr env]
     :symbol [(lookup expr env) env]
     :nil [expr env]
     :call (p-eval-call expr env)
     :let (p-eval-let expr env))))


;; Need to specialize functions. Need to handle
;; case. Need to have a nice way of documenting things.


(p-eval (analyze '(let* [x 2] (let* [y 3] (+ (+ x y) z)))))

(collect-defs (analyze-many '((def x 2) (def x (fn [x] (* x 2)))) analyze) {})
