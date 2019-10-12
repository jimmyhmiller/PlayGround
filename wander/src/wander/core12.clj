(ns wander.core12
  (:require [clojure.spec.alpha :as s]))


;; Specs
;; ---------------------------------------------------------------------

(s/def ::op
  keyword?)

(s/def ::ir
  (s/keys :req-un [::op]))

(s/def ::code
  any?)

(s/def ::symbol
  simple-symbol?)


;; Interfaces
;; ---------------------------------------------------------------------

(defn compile*-dispatch
  [ir fail-ir env]
  (get ir :op))

(defmulti compile*
  {:arglists '([ir fail-ir env])}
  #'compile*-dispatch
  :default ::default)

(defmulti children
  {:arglists '([ir])}
  (fn [ir] (get ir :op))
  :default ::default)

(defmethod children ::default
  [ir]
  [])

(defmulti make-node
  {:arglists '([ir children])}
  (fn [ir children]
    (get ir :op))
  :default ::default)

(defmethod make-node ::default
  [ir children]
  ir)

;; API
;; ---------------------------------------------------------------------

(defn fold
  [f init ir]
  (transduce
   (mapcat
    (fn [child]
      (cons child (children child))))
   f
   (f init ir)
   (children ir)))

(s/fdef fold
  :args (s/cat :f ifn? :val any? :ir ::ir)
  :ret any?)

(defn fmap
  [f ir]
  (make-node ir (map f (children ir))))

(defn prewalk [f ir]
  (let [ir* (f ir)]
    (if (reduced? ir*)
      (unreduced ir*)
      (fmap (fn [ir]
              (prewalk f ir))
            ir*))))

(defn postwalk [f ir]
  (f (fmap (fn [ir]
             (postwalk f ir))
           ir)))

(defn accumulate
  {:style/indent :defn}
  [expr-ir then-ir-fn]
  (let [symbol (gensym "X__")
        then-ir (then-ir-fn symbol)]
    {:op ::bind
     :symbol symbol
     :expr expr-ir
     :then then-ir}))

(defn bind
  {:style/indent :defn}
  [symbol expr-ir then-ir]
  {:op ::bind
   :symbol symbol
   :expr expr-ir
   :then then-ir})

(defn branch
  {:style/indent :defn}
  [symbol then-ir else-ir]
  {:op ::branch
   :symbol symbol
   :then then-ir
   :else else-ir})

(defn code
  [any]
  {:op ::code
   :code any})

(defn check-bounds
  {:style/indent :defn}
  [symbol length-ir]
  {:op ::check-bounds
   :symbol symbol
   :length length-ir})

(defn check-vector
  {:style/indent :defn}
  [symbol]
  {:op ::check-vector
   :symbol symbol})

(defn check-equal
  [symbol-1 symbol-2]
  {:op ::check-equal
   :symbol-1 symbol-1
   :symbol-2 symbol-2})

(defn map-get [symbol key-ir]
  {:op ::map-get
   :symbol symbol
   :key key-ir})

(defn nth-get [symbol index-ir]
  {:op ::nth-get
   :symbol symbol
   :index index-ir})

(defn return
  [code]
  {:op ::return
   :code code})

;; Implementation
;; ---------------------------------------------------------------------

(defmethod children ::bind
  [ir]
  [(get ir :expr)
   (get ir :then)])

(defmethod compile* ::bind
  [ir fail-ir env]
  (loop [ir ir
         bindings []]
    (let [bindings (conj bindings (get ir :symbol) (compile* (get ir :expr) fail-ir env))
          then-ir (get ir :then)]
      (case (get then-ir :op)
        ::bind
        (recur then-ir bindings)

        `(let ~bindings
           ~(compile* then-ir fail-ir env)))))
  #_
  `(let [~(get ir :symbol) ~(compile* (get ir :expr) fail-ir env)]
     ;; Put :symbol into environment.
     ~(compile* (get ir :then) fail-ir env)))

(defmethod make-node ::bind
  [ir [expr then]]
  (merge ir {:expr expr
             :then then}))

(defmethod children ::branch
  [ir]
  [(get ir :then)
   (get ir :else)])

(defmethod compile* ::branch
  [ir fail-ir env]
  `(if ~(get ir :symbol)
     ~(compile* (get ir :then) fail-ir env)
     ~(compile* (get ir :else) fail-ir env)))

(defmethod make-node ::branch
  [ir [then else]]
  (merge ir {:then then
             :else else}))

(defmethod compile* ::check-bounds
  [ir fail-ir env]
  `(= (count ~(get ir :symbol))
      ~(compile* (get ir :length) fail-ir env)))

(defmethod compile* ::check-map
  [ir fail-ir env]
  `(map? ~(get ir :symbol)))

(defmethod compile* ::check-vector
  [ir fail-ir env]
  `(vector? ~(get ir :symbol)))

(defmethod compile* ::check-equal
  [ir fail-ir env]
  `(= ~(get ir :symbol-1)
      ~(get ir :symbol-2)))

(defmethod compile* ::code
  [ir fail-ir env]
  (get ir :code))


(defmethod compile* ::nth-get
  [ir fail-ir env]
  `(nth ~(get ir :symbol) ~(compile* (get ir :index) fail-ir env))) 

(defmethod compile* ::return
  [ir fail-ir env]
  (let [code (get ir :code)]
    (case (get env ::match-type)
      :search
      `(list ~code)

      ;; else
      code)))





(defn propagate-compile-information-dispatch [{:keys [node fail env]}]
  (:op node))

(defmulti propagate-compile-information
  {:arglists '([ir fail-ir env])}
  #'propagate-compile-information-dispatch
  :default ::default)




(defn add-current-symbol [context]
  (assoc-in context [:env :parent-info :symbol] (get-in context [:node :symbol])))

(defn compile-subexpr [context expr-key]
  (let [compiled (propagate-compile-information
                  (assoc context :node (get-in context [:node expr-key])))]
    (-> context
        (assoc-in [:node expr-key] (:node compiled))
        (assoc-in [:env] (:env compiled)))))

(defn extract-subexpr [context expr-key]
  (assoc-in context [:node] (get-in context [:node expr-key])))

(defn get-current-symbol [context]
   (get-in context [:env :parent-info :symbol]))

(defn add-type-info [context code]
   ;; Need to handle s-expr and symbols properly
  (assoc-in context [:env :types (get-current-symbol context)] (type code)))

(defn get-type [context symbol]
  (get-in context [:env :types symbol]))

(defn add-additional-var-info [context value]
  ;; Need to handle s-expr properly
  (-> (if (coll? value)
        (assoc-in context [:env :size-info (get-current-symbol context)] (count value))
        context)
      (assoc-in [:env :value-info (get-current-symbol context)] value)))

(defn remove-node-if-unused [context]
  (if (get-in context [:env :unused-vars (get-in context [:node :symbol])])
    (assoc-in context [:node] (get-in context [:node :then]))
    context))

(defn mark-symbol-unused [context]
  (assoc-in context [:env :unused-vars (get-current-symbol context)] true))

(defn add-symbol-value [context value]
  (assoc-in context [:env :value-info (get-current-symbol context)] value))

(defn get-var-value [context symbol]
   (get-in context [:env :value-info symbol]))

(defn get-var-size [context symbol]
   (get-in context [:env :size-info symbol]))

(defn change-node [context new-node]
  (assoc-in context [:node] new-node))

(defmethod propagate-compile-information ::bind [{:keys [node fail env] :as context}]
  (-> context
      (add-current-symbol)
      (compile-subexpr :expr)
      (compile-subexpr :then)
      (remove-node-if-unused)))

(defmethod propagate-compile-information ::code [{:keys [node fail env] :as context}]
  (let [code (:code node)]
    (-> context
        (add-type-info  code)
        (add-additional-var-info code))))


(defmethod propagate-compile-information ::check-vector [{:keys [node fail env] :as context}]
  (let [current-type (get-type context (:symbol node))]
    (cond
      (nil? current-type)
      context
      (isa? current-type clojure.lang.IPersistentVector)
      (-> context
          (mark-symbol-unused)
          (add-symbol-value true))
      :else
      (-> context 
          (mark-symbol-unused)
          (add-symbol-value false)))))

(defmethod propagate-compile-information ::branch [{:keys [node fail env] :as context}]
   (let [value (get-var-value context (:symbol node))]
    (cond
      (nil? value)
      (-> context
          (compile-subexpr :then)
          (compile-subexpr :else))
      (true? value)
      (-> context
          (compile-subexpr :then)
          (extract-subexpr :then)) 
      :else
      (-> context
          (compile-subexpr :else)
          (extract-subexpr :else)))))


(defmethod propagate-compile-information ::check-bounds [{:keys [node fail env] :as context}]
   (let [size (get-var-size context (:symbol node))
         expected-size (get-in node [:length :code])]
     (cond
       (nil? size)
       context
       (= size expected-size size)
       (-> context
           (mark-symbol-unused)
           (add-symbol-value true))
       :else
       (-> context 
           (mark-symbol-unused)
           (add-symbol-value false)))))


(defmethod propagate-compile-information ::nth-get [{:keys [node fail env] :as context}]
  (let [value (get-var-value context (:symbol node))
        index (get-in node [:index :code])]
    (println value index)
    (if (nil? value) 
      context
      (-> context
          (add-symbol-value value)
          (change-node (code value))))))


(defmethod propagate-compile-information ::check-equal [{:keys [node fail env] :as context}]
  context)

(defmethod propagate-compile-information ::check-equal [{:keys [symbol-1 symbol-2] :as node} fail-ir env]
  (let [current-symbol (:current-symbol env)
        symbol-1-value (get-in env [:value-info symbol-1] :not-found)
        symbol-2-value (get-in env [:value-info symbol-2] :not-found)]
    (cond
      (or (= symbol-1-value :not-found) (= symbol-2-value :not-found))
      {:env env
       :node node}
      (= symbol-1-value symbol-2-value)
      {:env (-> env 
                (assoc-in [:value-info current-symbol] true)
                (assoc-in [:unused-vars current-symbol] true)
                (assoc-in [:unused-vars symbol-1] true)
                (assoc-in [:unused-vars symbol-2] true))
       :node node}
      :else

      {:env (-> env
                (assoc-in [:value-info current-symbol] false)
                (assoc-in [:unused-vars current-symbol] true)
                (assoc-in [:unused-vars symbol-1] true)
                (assoc-in [:unused-vars symbol-2] true))
       :node node})))

(defmethod propagate-compile-information ::return [{:keys [node fail env] :as context}]
  context)




(propagate-compile-information {:node example :fail :fail :env {}})

(compile*
 (:node (propagate-compile-information example :fail {})) 
 (code nil)
 {})

(def example
  (accumulate (code [1])
    (fn [target]
      (accumulate (check-vector target)
        (fn [bool]
          (branch bool
            (accumulate (check-bounds target (code 1))
              (fn [bool]
                (branch bool
                  (accumulate (nth-get target (code 0))
                    (fn [nth0]
                      (accumulate (code 1)
                        (fn [val0]
                          (accumulate (check-equal nth0 val0)
                            (fn [bool]
                              (branch bool
                                (return true)
                                (return false))))))))
                  (code nil))))
            (code nil)))))))


