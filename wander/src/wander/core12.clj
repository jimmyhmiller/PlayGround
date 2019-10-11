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


(defmulti propagate-compile-information
  {:arglists '([ir fail-ir env])}
  #'compile*-dispatch
  :default ::default)

(defmethod propagate-compile-information ::bind [{:keys [symbol expr then] :as node} fail-ir env]
  (let [expr-info (propagate-compile-information expr fail-ir (assoc env :current-symbol symbol))
        env (:env expr-info)
        then-info (propagate-compile-information then fail-ir env)
        more-data (if (= (:node then-info) then) 
                    then-info
                    (propagate-compile-information (:node then-info) fail-ir (:env then-info)))]
    {:env (:env more-data)
     :node (if
             (get-in (:env more-data) [:unused-vars symbol]) (:node more-data)
             (assoc node
                    :expr (:node expr-info)
                    :then (:node more-data)))}))

(defmethod propagate-compile-information ::code [{:keys [code] :as node} fail-ir env]
  (let [new-env (if (coll? code)
                  ;; have handle s-expressions and symbols
                  (-> env
                      (assoc-in [:type-info (:current-symbol env)] (type code))
                      (assoc-in [:size-info (:current-symbol env)] (count code))
                      (assoc-in [:value-info (:current-symbol env)] code))
                  (-> env
                      (assoc-in [:type-info (:current-symbol env)] (type code))
                       (assoc-in [:value-info (:current-symbol env)] code)))]
    {:env new-env
     :node node}))


(defmethod propagate-compile-information ::check-vector [{:keys [symbol] :as node} fail-ir env]
  (let [current-symbol (:current-symbol env)
        type-of-symbol (get-in env [:type-info symbol])]
    (cond
      (not type-of-symbol)
      {:env env
       :node node}
      (isa? type-of-symbol clojure.lang.IPersistentVector)
      {:env (-> env
                (assoc-in [:value-info current-symbol] true)
                (assoc-in [:unused-vars current-symbol] true))
       :node node}
      :else
      {:env (-> env
                (assoc-in [:value-info current-symbol] false)
                (assoc-in [:unused-vars current-symbol] true))
       :node node})))

(defmethod propagate-compile-information ::branch [{:keys [symbol then else] :as node} fail-ir env]
  (let [value-of-symbol (get-in env [:value-info symbol])]
    (cond
      (nil? value-of-symbol)
      {:env env
       :node node}
      (true? value-of-symbol)
      {:env env
       :node then}
      :else
      {:env (assoc-in env [:value-info symbol] false)
       :node else})))


(defmethod propagate-compile-information ::check-bounds [{:keys [symbol length] :as node} fail-ir env]
  (let [{:keys [env]} (propagate-compile-information length fail-ir env)
        current-symbol (:current-symbol env)
        size-of-symbol (get-in env [:size-info symbol])
        length (get-in length [:code])]
    (cond
      (nil? size-of-symbol)
      {:env env
       :node node}
      (= size-of-symbol length)
      {:env (-> env 
                (assoc-in [:value-info current-symbol] true)
                (assoc-in [:unused-vars current-symbol] true))
       :node node}
      :else
      {:env (-> env
                (assoc-in [:value-info current-symbol] false)
                (assoc-in [:unused-vars current-symbol] true))
       :node node})))

(defmethod propagate-compile-information ::nth-get [{:keys [index symbol then] :as node} fail-ir env]
  (let [current-symbol (:current-symbol env)
        value-of-symbol (get-in env [:value-info symbol])
        index (get-in index [:code])]
    (if (nil? value-of-symbol)
      {:env env
       :node node}
      (let [value (nth value-of-symbol index)]
        {:env (assoc-in env [:value-info current-symbol] value)
         :node (code value)}))))

(defmethod propagate-compile-information ::check-equal [{:keys [symbol-1 symbol-2] :as node} fail-ir env]
  (let [current-symbol (:current-symbol env)
        symbol-1-value (get-in env [:value-info symbol-1] :not-found)
        symbol-2-value (get-in env [:value-info symbol-2] :not-found)]
    (cond
      (or (=  symbol-1-value :not-found) (= symbol-2-value :not-found))
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

(defmethod propagate-compile-information ::return [{:keys [] :as node} fail-ir env]
  {:env env
   :node node})


(propagate-compile-information example :fail {})

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

;; Scratch
;; ---------------------------------------------------------------------

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
 {})
;; =>
(let [X__8771 [1]
      X__8772 (vector? X__8771)]
  (if X__8772
    (let [X__8773 (= (count X__8771) 1)]
      (if X__8773
        (let [X__8774 (nth X__8771 0)
              X__8775 1
              X__8776 (= X__8774 X__8775)]
          (if X__8776 true false))
        nil))
    nil))
