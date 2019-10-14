(ns wander.core13
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
  [symbol length-ir]
  {:op ::check-bounds
   :symbol symbol
   :length length-ir})

(defn check-equal
  [symbol-1 symbol-2]
  {:op ::check-equal
   :symbol-1 symbol-1
   :symbol-2 symbol-2})

(defn check-fail
  [symbol]
  {:op ::check-fail
   :symbol symbol})

(defn check-map
  [symbol]
  {:op ::check-map
   :symbol symbol})

(defn check-set
  [symbol]
  {:op ::check-set
   :symbol symbol})

(defn check-seq
  [symbol]
  {:op ::check-seq
   :symbol symbol})

(defn check-vector
  [symbol]
  {:op ::check-vector
   :symbol symbol})

(def fail
  {:op ::fail})

(defn logic-variable-bind
  {:style/indent :defn}
  [logic-variable-symbol expr-ir then-ir]
  {:op ::logic-variable-bind
   :logic-variable-symbol logic-variable-symbol
   :expr expr-ir
   :then then-ir})

(defn logic-variable-check
  [logic-variable-symbol symbol]
  {:op ::logic-variable-check
   :logic-variable-symbol logic-variable-symbol
   :symbol symbol})

(defn map-get [symbol key-ir]
  {:op ::map-get
   :symbol symbol
   :key key-ir})

(defn memory-variable-init
  {:style/indent :defn}
  [memory-variable-symbol then-ir]
  {:op ::memory-variable-init
   :memory-variable-symbol memory-variable-symbol
   :then then-ir})

(defn memory-variable-append
  {:style/indent :defn}
  [memory-variable-symbol symbol then-ir]
  {:op ::memory-variable-append
   :memory-variable-symbol memory-variable-symbol
   :symbol symbol
   :then then-ir})

(defn nth-get [symbol index-ir]
  {:op ::nth-get
   :symbol symbol
   :index index-ir})

(defn return
  [code]
  {:op ::return
   :code code})

(defn star
  {:style/indent :defn}
  [symbol n return-symbols body-ir-fn then-ir-fn]
  (let [input (gensym "I__")
        variables (vec return-symbols)]
    {:op ::star
     ;; The symbol of the sequence to process.
     :symbol symbol
     ;; The symbol passed to the body.
     :input input
     :n n
     :return-symbols variables
     :body (body-ir-fn input variables (return variables))
     :then (then-ir-fn input variables)}))

(defn const
  [any]
  {:op :const
   :const any})

(defn define
  {:style/indent :defn}
  [requires returns body-ir-fn then-ir-fn]
  (let [requires (vec requires)
        returns (vec returns)
        symbol (gensym "S__")
        input (gensym "I__")]
    {:op ::define
     :symbol symbol
     :input input
     :requires requires
     :returns returns
     :body (body-ir-fn symbol input requires (return returns))
     :then (then-ir-fn symbol requires)}))

(defn call [symbol input requires]
  {:op ::call
   :symbol symbol
   :input input
   :requires requires})

;; Implementation
;; ---------------------------------------------------------------------

(defmethod children ::bind
  [ir]
  [(get ir :expr)
   (get ir :then)])

(defmethod compile* ::bind
  [ir fail-ir env]
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

(defmethod compile* ::call
  [ir fail-ir env]
  `(~(get ir :symbol) ~(get ir :input) ~@(get ir :requires)))

(defmethod compile* ::check-fail
  [ir fail-ir env]
  `(= ::fail ~(get ir :symbol)))


(defmethod compile* ::check-bounds
  [ir fail-ir env]
  `(= (count ~(get ir :symbol))
      ~(compile* (get ir :length) fail-ir env)))

(defmethod compile* ::check-equal
  [ir fail-ir env]
  `(= ~(get ir :symbol-1)
      ~(get ir :symbol-2)))

(defmethod compile* ::check-logic-variable
  [ir fail-ir env]
  `(= ~(get ir :symbol-1)
      ~(get ir :symbol-2)))

(defmethod compile* ::check-map
  [ir fail-ir env]
  `(map? ~(get ir :symbol)))

(defmethod compile* ::check-set
  [ir fail-ir env]
  `(set? ~(get ir :symbol)))

(defmethod compile* ::check-seq
  [ir fail-ir env]
  `(seq? ~(get ir :symbol)))

(defmethod compile* ::check-vector
  [ir fail-ir env]
  `(vector? ~(get ir :symbol)))

(defmethod compile* ::code
  [ir fail-ir env]
  (get ir :code))

(defmethod compile* ::define
  [ir fail-ir env]
  `(letfn [(~(get ir :symbol) [~(get ir :input) ~@(get ir :requires)]
            ~(compile* (get ir :body) fail-ir env))]
     ~(compile* (get ir :then) fail-ir env)))

(defmethod compile* ::fail
  [ir fail-ir env]
  (compile* fail-ir fail-ir env))

(defmethod compile* ::logic-variable-bind
  [ir fail-ir env]
  `(let [~(get ir :symbol) ~(compile* (get ir :expr) fail-ir env)]
     ;; Put :symbol into environment.
     ~(compile* (get ir :then) fail-ir env)))

(defmethod compile* ::nth-get
  [ir fail-ir env]
  `(nth ~(get ir :symbol) ~(compile* (get ir :index) fail-ir env))) 

(defmethod compile* ::memory-variable-append
  [ir fail-ir env]
  `(let [~(get ir :memory-variable-symbol) (conj ~(get ir :memory-variable-symbol) ~(get ir :symbol))] 
     ~(compile* (get ir :then) fail-ir env)))

(defmethod compile* ::memory-variable-init
  [ir fail-ir env]
  `(let [~(get ir :memory-variable-symbol) []]
     ;; Put :symbol into environment.
     ~(compile* (get ir :then) fail-ir env)))

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

(defn remove-current-symbol [context]
  (update-in context [:env :parent-info] dissoc :symbol))

(defn add-current-function [context]
  (assoc-in context [:env :parent-info :function] (get-in context [:node :symbol])))

(defn remove-current-function [context]
  (update-in context [:env :parent-info] dissoc :function))

(defn get-current-function [context]
  (get-in context [:env :parent-info :function]))

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

(defn get-type [context symbol default-value]
  (get-in context [:env :types symbol] default-value))

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

(defn mark-symbol-unused 
  ([context]
   (mark-symbol-unused context (get-current-symbol context)))
  ([context symbol]
   (assoc-in context [:env :unused-vars symbol] true)))

(defn add-symbol-value [context value]
  (assoc-in context [:env :value-info (get-current-symbol context)] value))

(defn get-var-value [context symbol default-value]
  (get-in context [:env :value-info symbol] default-value))

(defn get-var-size [context symbol default-value]
   (get-in context [:env :size-info symbol] default-value))

(defn change-node [context new-node]
  (assoc-in context [:node] new-node))

(defn add-function-to-context [context]
  (assoc-in context [:env :functions (get-in context [:node :symbol]) :node]
            (dissoc (get-in context [:node]) :then)))

(defn add-function-can-fail [context]
  (when-let [current-function (get-current-function context)]
    (assoc-in context [:env :functions current-function :can-fail] true)))

(defn add-call-information)

(defmethod propagate-compile-information ::bind [{:keys [node fail env] :as context}]
  (-> context
      (add-current-symbol)
      (compile-subexpr :expr)
      (remove-current-symbol) 
      (compile-subexpr :then)
      (remove-node-if-unused)))

(defmethod propagate-compile-information ::code [{:keys [node fail env] :as context}]
  (let [code (:code node)]
    (-> context
        (add-type-info  code)
        (add-additional-var-info code))))

(defmethod propagate-compile-information ::check-vector [{:keys [node fail env] :as context}]
  (let [current-type (get-type context (:symbol node) ::not-found)]
    (cond
      (= current-type ::not-found)
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
  (let [value (get-var-value context (:symbol node) ::not-found)]
    (cond
      (= value ::not-found)
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
  (let [size (get-var-size context (:symbol node) ::not-found)
        expected-size (get-in node [:length :code])]
    (cond
      (= size ::not-found)
      context
      (>= size expected-size)
      (-> context
          (mark-symbol-unused)
          (add-symbol-value true))
      :else
      (-> context 
          (mark-symbol-unused)
          (add-symbol-value false)))))


(defmethod propagate-compile-information ::nth-get [{:keys [node fail env] :as context}]
  (let [value (get-var-value context (:symbol node) ::not-found)
        index (get-in node [:index :code])]
    (if (= value ::not-found)
      context
      (-> context
          (add-symbol-value (nth value index))
          (change-node (code (nth value index)))))))


(defmethod propagate-compile-information ::check-equal [{:keys [node fail env] :as context}]
  (let [symbol-1 (:symbol-1 node)
        symbol-2 (:symbol-2 node)
        symbol-1-value (get-var-value context symbol-1 ::not-found)
        symbol-2-value (get-var-value context symbol-2 ::not-found)]
    (cond
      (or (= symbol-1-value ::not-found)
          (= symbol-2-value ::not-found))
      context
      (= symbol-1-value symbol-2-value)
      (-> context
          (add-symbol-value true)
          (mark-symbol-unused)
          (mark-symbol-unused symbol-1)
          (mark-symbol-unused symbol-2))
      :else
      (-> context
          (add-symbol-value false)
          (mark-symbol-unused)
          (mark-symbol-unused symbol-1)
          (mark-symbol-unused symbol-2)))))

(defmethod propagate-compile-information ::return [{:keys [node fail env] :as context}]
  context)

(defmethod propagate-compile-information ::star [{:keys [node fail env] :as context}]
  context)

(defmethod propagate-compile-information ::define [{:keys [node fail env] :as context}]
  (-> context
      (add-current-function)
      (compile-subexpr :body)
      (remove-current-function)
      (add-function-to-context)
      (compile-subexpr :then)))


(defmethod propagate-compile-information ::call [{:keys [node fail env] :as context}]
  (-> context
      (compile-subexpr :then)))

(defmethod propagate-compile-information ::check-fail [{:keys [node fail env] :as context}]
  context)


(defmethod propagate-compile-information ::fail [{:keys [node fail env] :as context}]
  (-> context
      (add-function-can-fail)))

(defmethod propagate-compile-information ::memory-variable-append [{:keys [node fail env] :as context}]
  context)


(defmethod propagate-compile-information ::memory-variable-init [{:keys [node fail env] :as context}]
  (-> context
      (compile-subexpr :then)))

(propagate-compile-information {:node example2 :fail :fail :env {}})



(def example2
  (define '[!xs !ys] '[!xs !ys]
     (fn body [symbol input [!xs !ys] return-ir]
       (accumulate (nth-get input (code 0))
         (fn [x]
           (accumulate (nth-get input (code 1))
             (fn [y]
               (memory-variable-append !xs x
                 (memory-variable-append !ys y
                   return-ir)))))))
     (fn then [symbol [!xs !ys :as variables]]
       (memory-variable-init '!xs
         (memory-variable-init '!ys
           (accumulate (code ["x" "y"])
             (fn [val]
               (accumulate (call symbol val variables)
                 (fn [state]
                   (accumulate (check-fail state)
                     (fn [fail?]
                       (branch fail?
                         {:op ::fail}
                         (bind !xs (nth-get state (code 0))
                           (bind !ys (nth-get state (code 1))
                             (return variables)))))))))))))))

(compile*
 (:node (propagate-compile-information
         '{:fail :fail
           :env {}
           :node
           {:op ::bind
            :symbol I__12981
            :expr {:op ::code :code ["x" "y"]}
            :then
            {:op :wander.core13/bind,
             :symbol X__12982,
             :expr {:op :wander.core13/nth-get, :symbol I__12981, :index {:op ::code :code 0}},
             :then
             {:op :wander.core13/bind,
              :symbol X__12983,
              :expr {:op :wander.core13/nth-get, :symbol I__12981, :index {:op ::code :code 1}},
              :then
              {:op :wander.core13/memory-variable-append,
               :memory-variable-symbol !xs,
               :symbol X__12982,
               :then
               {:op :wander.core13/memory-variable-append,
                :memory-variable-symbol !ys,
                :symbol X__12983,
                :then {:op :wander.core13/return, :code [!xs !ys]}}}}}}}))
 (code nil)
 {})



(compile* example2 (code nil) {})


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


;; Scratch
;; ---------------------------------------------------------------------

(comment
  (compile*
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
             (code nil))))))
   (code nil)
   {}))

(comment
  (accumulate (code [1 2 3 4])
    (fn [xs]
      (star xs 2 '[!xs !ys]
        (fn [xs [!xs !ys] return-ir]
          (accumulate (nth-get xs (code 0))
            (fn [nth0]
              (accumulate (nth-get xs (code 0))
                (fn [nth1]
                  (memory-variable-append !xs nth0
                    (memory-variable-append !ys nth1
                      return-ir)))))))
        (fn [tail variables]
          (return variables))))))



(comment
  (compile*
   ;; !xs !ys => !xs !ys
   (define '[!xs !ys] '[!xs !ys]
     (fn body [symbol input [!xs !ys] return-ir]
       (accumulate (code "x")
         (fn [x]
           (accumulate (code "y")
             (fn [y]
               (memory-variable-append !xs x
                 (memory-variable-append !ys y
                   return-ir)))))))
     (fn then [symbol [!xs !ys :as variables]]
       (memory-variable-init '!xs
         (memory-variable-init '!ys
           (accumulate (code ["x" "y"])
             (fn [val]
               (accumulate (call symbol val variables)
                 (fn [state]
                   (accumulate (check-fail state)
                     (fn [fail?]
                       (branch fail?
                         {:op ::fail}
                         (bind !xs (nth-get state (code 0))
                           (bind !ys (nth-get state (code 1))
                             (return variables))))))))))))))
   (code nil)
   {})
)
