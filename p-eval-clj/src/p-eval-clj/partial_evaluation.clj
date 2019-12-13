(ns p-eval-clj.partial-evaluation
  (:require [meander.epsilon :as m]
            [meander.syntax.zeta :as zyntax]))



(defn atom? [expr]
  
  (cond 
    (and (seq? expr) (= (first expr) 'quote))
    true
    (coll? expr)
    (every? atom? expr)
    :else (boolean
           ((some-fn number? string? fn? keyword? boolean? nil?) expr))))


(defn collection-literal? [expr]
  (m/match expr
    (m/or [& _]
          ((m/or list clojure.core/list) & _)
          {}
          #{})
    true
    _ false))

(defn split-pred [pred coll]
  (let [grouped (group-by pred coll)]
    [(get grouped true) (get grouped false)]))

(defn with-args [env arg-names arg-values]
  (into env (mapv vector arg-names arg-values)))

(defn specialize-fn-name [function-name atom-args]
  (symbol (str function-name (if atom-args (hash atom-args) ""))))



(defmulti specialize (fn [env globals expr] (first expr)))


(defn evaluate* [env globals expr]
  (let [result
        (m/match expr

          globals @globals

          (m/pred atom? ?x) ?x

          (m/pred symbol? ?x)
          (cond (contains? env ?x)
                (get env ?x)
                
                (and (contains? @globals ?x)
                     (contains? (get @globals ?x) :value))
                (get-in @globals [?x :value])
                :else ?x)

          (def ?name ?value)
          (swap! globals assoc ?name {:value ?value})

          (defn ?name ?args ?body)
          (swap! globals assoc ?name {:args ?args
                                      :body ?body})


          (let [?var ?val] ?body)
          (let [evaled-val (evaluate* env globals ?val)
                renamed-var (gensym)]
            (if (atom? evaled-val)
              (evaluate* (assoc env ?var evaled-val) globals ?body)
              (let [new-env (if (collection-literal? evaled-val)
                              (assoc env ?var evaled-val)
                              (assoc env ?var renamed-var))]
                `(~'let [~renamed-var ~evaled-val]
                  ~(evaluate* new-env globals ?body)))))

          ((m/or cond clojure.core/cond))
          nil

          ((m/or cond clojure.core/cond) & _ :as ?cond)
          (evaluate* env globals (macroexpand ?cond))

          ;; This doesn't work as a literal (?args) so we make sure to preserve the list
          ((m/or list clojure.core/list) & ?args)
          `(clojure.core/list ~@(mapv (partial evaluate* env globals) ?args))

          (if ?pred ?t)
          (evaluate* env globals (list 'if ?pred ?t nil))

          (if ?pred ?t ?f)
          (let [evaled-pred (evaluate* env globals ?pred)]
            (cond 
              (atom? evaled-pred)
              (if evaled-pred
                (evaluate* env globals ?t)
                (evaluate* env globals ?f))

              (collection-literal? evaled-pred)
              (evaluate* env globals ?t)

              :else 
              `(if ~evaled-pred
                 ~(evaluate* env globals ?t)
                 ~(evaluate* env globals ?f))))


          (fn ?args ?body)
          ;; singlular body for now
          (let [renamed-args (mapv (fn [_] (gensym)) ?args)
                evaled-body (evaluate* (into env (mapv vector ?args renamed-args)) globals ?body)]
            (if (atom? evaled-body)
              ;; (Not sure I love this)
              (eval `(fn [~@renamed-args] ~evaled-body))
              `(~'fn [~@renamed-args] ~evaled-body)))



          ((m/and (m/symbol "clojure.core" _) ?name) & ?args :as ?expr)
          (let [e-args (mapv (partial evaluate* env globals) ?args)]
            (if (every? atom? e-args)
              (apply (resolve ?name) e-args)
              ;; add the ability to specialize here?
              (if-let [special-method (get-method specialize ?name)]
                (special-method env globals (cons ?name e-args))
                (cons ?name e-args))))
          
          (?name & ?args)
          (if-let [{:keys [args body]} (get @globals ?name)]
            (let [e-args (mapv (partial evaluate* env globals) ?args)
                  [atom-args not-atom-args] (split-pred (comp atom? second) (mapv vector args e-args))]
              (if (empty? not-atom-args)
                ;; All our args are atoms, so let's eval now
                (evaluate* (into env atom-args) globals body)
                (let [new-fn-name (specialize-fn-name ?name atom-args)]
                  (if (not (contains? @globals new-fn-name))
                    (do
                      ;; Add temporary placehold to make sure in recursion
                      ;; we don't try to do this again
                      (swap! globals assoc new-fn-name {:current true
                                                        :args nil
                                                        :body nil})
                      (let [renamed-atom-args (mapv (fn [_] (gensym)) atom-args)
                            renamed-not-atom-args (mapv (fn [_] (gensym)) not-atom-args)
                            evaled-body (evaluate* 
                                         (into env (mapv vector (mapv first atom-args) renamed-atom-args))
                                         globals body)
                            specialized (evaluate* 
                                         (into env (mapv vector renamed-atom-args (mapv second atom-args)))
                                         globals evaled-body)
                            extra-special (evaluate* (into env (mapv vector 
                                                                     (mapv first not-atom-args) 
                                                                     renamed-not-atom-args))
                                                     globals 
                                                     specialized)]
                        (swap! globals assoc new-fn-name {:current false
                                                          :args renamed-not-atom-args
                                                          :body extra-special})))
                    (swap! globals assoc-in [new-fn-name :current] true))
                  
                  ;; This is ugly. We do need an inline policy. But perhaps better?
                  (let [{:keys [args body current]} (get @globals new-fn-name)
                        result (if (and current (not-empty (filter #{new-fn-name} (tree-seq coll? seq body))))
                                 `(~new-fn-name ~@(mapv second  not-atom-args))
                                 (evaluate* (into env (mapv vector args 
                                                            (mapv second not-atom-args))) globals body))]
                    (swap! globals assoc-in [new-fn-name :current] false)
                    result))))
            
            (let [resolved (resolve ?name)]
              (if resolved
                (evaluate* env globals `(~(symbol resolved) 
                                         ~@(mapv (partial evaluate* env globals) ?args)))
                (throw (ex-info "no resolved" {:name ?name})))))



          (m/pred vector? ?x) (mapv (partial evaluate* env globals) ?x)
          (m/pred seq? ?x) (doall (map (partial evaluate* env globals) ?x))
          (m/pred map? ?x) (reduce-kv (fn [acc k v] (assoc acc
                                                           (evaluate* env globals k)
                                                           (evaluate* env globals v))) {} ?x)
          ?x (throw (ex-info "not found" {:expr ?x})))]

    #_(println expr " => " result)
    result))


(defmethod specialize 'clojure.core/mapcat [env globals expr]
  ;; handle single case right now
  (m/match expr
    (_ (fn [?arg] ?body) (clojure.core/list ?x))
    (evaluate* (assoc env ?arg (evaluate* env globals ?x)) globals ?body)
    
    _ expr))

(defmethod specialize 'clojure.core/empty? [env globals expr]
  (m/match expr
    (_ [_ & _]) false
    _ expr))


(defmethod specialize 'clojure.core/first [env globals expr]
  (m/match expr
    (_ [?x & _]) ?x
    _ expr))


(defmethod specialize 'clojure.core/ffirst [env globals expr]
  (m/match expr
    (_ [[?x & _] & _]) ?x
    _ expr))


(defmethod specialize 'clojure.core/conj [env globals expr]
  (m/match expr
    (_ [& ?coll] ?x) (conj ?coll ?x)
    _ expr))


(defmethod specialize 'clojure.core/vector? [env globals expr]
  (m/match expr
    (_ [& _]) true
    _ expr))

(defmethod specialize 'clojure.core/seq [env globals expr]
  (println "here" expr env)
  (m/match expr
    (_ [& ?coll]) `(clojure.core/list ~@(seq ?coll))
    _ expr))

(defmethod specialize 'clojure.core/next [env globals expr]
  (m/match expr
    (_ [_ & ?coll]) ?coll
    _ expr))


(defmethod specialize 'clojure.core/nth [env globals expr]
  (m/match expr
    (_ [& ?coll] ?n) (nth ?coll ?n)
    _ expr))







(defn evaluate [exprs]
  (let [globals (atom {})]
     (mapv (partial evaluate* {} globals) exprs)))



(last
 (evaluate 
  '[(defn insert [coll k v]
      (if (empty? coll)
        [[k v]]
        (conj coll [k v])))


    (defn lookup [coll x]
      (if (clojure.core/empty? coll)
        nil
        (if (clojure.core/= (clojure.core/ffirst coll) x)
          (clojure.core/second (clojure.core/first coll))
          (lookup (clojure.core/rest coll) x))))


    (defn interpret [ast form smap]
      (let [tag (clojure.core/get ast :tag)]

        (if (clojure.core/= tag :literal)
          (let [x (clojure.core/get ast :form)]
            (if (clojure.core/= x form)
              (clojure.core/list smap)))

          (if (clojure.core/= tag :logic-variable)
            (let [other-form (lookup smap ast)]
              (if other-form
                (if (clojure.core/= other-form form)
                  (clojure.core/list smap))
                (clojure.core/list (insert smap ast form))))

            (if (clojure.core/= tag :cat)
              (if (clojure.core/seq form)
                (let [next-one (clojure.core/next form)]
                  (clojure.core/mapcat
                   (fn [smap]
                     (interpret (clojure.core/get ast :next) next-one smap))
                   (interpret (clojure.core/get ast :pattern) (clojure.core/nth form 0) smap))))

              (if (clojure.core/= tag :vector)
                (if (clojure.core/vector? form)
                  (let [pattern (clojure.core/get ast :pattern)]
                    (interpret pattern form smap)))

                (if (clojure.core/= tag :empty)
                  (if (clojure.core/seq form)
                    nil
                    (clojure.core/list smap))
                  ast)))))))


    (interpret 
     {:tag :vector,
      :pattern
      {:tag :cat,
       :pattern {:tag :logic-variable, :symbol '?x, :form '?x},
       :next
       {:tag :cat,
        :pattern
        {:tag :vector,
         :pattern
         {:tag :cat,
          :pattern {:tag :logic-variable, :symbol '?a, :form '?a},
          :next
          {:tag :cat,
           :pattern {:tag :logic-variable, :symbol '?b, :form '?b},
           :next
           {:tag :cat,
            :pattern {:tag :logic-variable, :symbol '?c, :form '?c},
            :next {:tag :empty}}}},
         :form ['?a '?b '?c]},
        :next
        {:tag :cat,
         :pattern {:tag :logic-variable, :symbol '?z, :form '?z},
         :next {:tag :empty}}}},
      :form '[?x [?a ?b] ?z]}
     [1 [a b] 2]
     [])
    
    ]))





(prn)
(zyntax/parse '[?x [?a ?b ?c] ?z])

