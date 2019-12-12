(ns p-eval-clj.core
  (:require [meander.epsilon :as m]
            [clojure.walk :as walk]
            [meander.syntax.zeta :as zyntax]
            [clojure.string :as string]))


(defn logic-variable? [x]
  (and (symbol? x)
       (string/starts-with? (name x) "?")))

(defn atom? [expr]
  (if
    (coll? expr)
    ;; this isn't right
    (every? (some-fn atom? logic-variable?) expr)
    (boolean
     ((some-fn number? string? fn? keyword? boolean? nil?) expr))))

(defn split-pred [pred coll]
  (let [grouped (group-by pred coll)]
    #_(println grouped)
    [(get grouped true) (get grouped false)]))

(defn map-strict [f & args]
  (doall (apply map f args)))

(defn peval2 [expr env fns]
  ;; get rid of this ugliness
  (let [fdef (atom fns)]
    (letfn [(peval2' [expr env]
            #_ (println expr)
              m/match expr

              (m/pred logic-variable?)
              `(quote ~expr)

              (m/and (m/pred symbol? ?x))
              (get env ?x ?x)


              ;; Need to split this stuff out into specialization
              ;; or even better figure out a way of partial evaluating
              ;; where we can be sure these things are safe even though
              ;; not everything here is an atom.
              (clojure.core/list & ?args)
              `(clojure.core/list ~@(map-strict #(peval2' % env) ?args))

              
              (clojure.core/mapcat (fn [?arg] ?body) (clojure.core/list ?x))
              (peval2' ?body (assoc env ?arg (peval2' ?x env)))

              
              (clojure.core/empty? ?coll)
              (let [result (peval2' ?coll env)]
                (if (coll? result)
                  (empty? result)
                  `(clojure.core/empty? ~result)))

              (clojure.core/first (m/pred coll? ?coll))
              (let [result (peval2' ?coll env)]
                (m/match result
                  [?k & _] (peval2' ?k env)
                  _ `(clojure.core/first ~result)))


              (clojure.core/second (m/pred coll? ?coll))
              (let [result (peval2' ?coll env)]
                (m/match result
                  [_ ?k & _] (peval2' ?k env)
                  _ `(clojure.core/second ~result)))
              
              
              
              (clojure.core/ffirst (m/pred coll? ?coll))
              (let [result (peval2' ?coll env)]
                (m/match result
                  [[?k _] & _] (peval2' ?k env)
                  _ `(clojure.core/ffirst ~result)))
              
              
              (clojure.core/conj [& _ :as ?args] ?arg)
              (let [coll-result (mapv #(peval2' % env) ?args)
                    arg (peval2' ?arg env)]
                (if (coll? coll-result)
                  (conj coll-result arg)
                  `(clojure.core/conj ~coll-result ~arg)))
              
              ((m/and (m/symbol "clojure.core" ?sym) ?f) & ?args :as ?expr)
              (do
                (if (= ?sym "ffirst") (println ?expr))
                (let [args (map-strict #(peval2' % env) ?args)]
                  (if (every? atom? args)
                    (apply (resolve ?f) args)
                    ;; better specializing
                    (if (and (#{"ffirst"} ?sym) (coll? (first args)))
                      (peval2'   (cons ?f args) env)
                      (cons ?f args)))))

              (let [?var ?val] ?body)
              (let [evaled-val (peval2' ?val env)
                    renamed-var (gensym)]
                (if (atom? evaled-val)
                  (peval2' ?body (assoc env ?var evaled-val))
                  `(~'let [~renamed-var ~evaled-val]
                    ~(peval2' ?body (assoc env ?var renamed-var)))))

              (if ?pred ?t)
              (peval2' (list 'if ?pred ?t nil) env)

              (if ?pred ?t ?f)
              (let [evaled-pred (peval2' ?pred env)]
                (if (atom? evaled-pred)
                  (if evaled-pred
                    (peval2' ?t env)
                    (peval2' ?f env))
                  `(if ~evaled-pred
                     ~(peval2' ?t env)
                     ~(peval2' ?f env))))

              (fn ?args ?body)
              ;; singlular body for now
              (let [renamed-args (map-strict (fn [_] (gensym)) ?args)
                    evaled-body (peval2' ?body (merge env (into {} (map vector ?args renamed-args))))]
                (if (atom? evaled-body)
                  (eval `(fn [~@renamed-args] ~evaled-body))
                  `(~'fn [~@renamed-args] ~evaled-body)))

              (?f & ?args :as ?expr)
              (if-let [{:keys [args body]} (get @fdef ?f)]
                (let [evaled-args (map-strict vector args (map-strict #(peval2' % env) ?args))
                      [atoms not-atoms] (split-pred (comp atom? second) evaled-args)]
                  (if (empty? not-atoms)
                    (peval2' body (merge env (into {} atoms)))
                    ;; make sure empty atoms are empty string
                    (let [f' (symbol (str ?f (if atoms (hash atoms) "")))]
                      (when (not (contains? @fdef f'))
                        (do (swap! fdef assoc f' {:args nil
                                                  :body nil})
                            (swap! fdef assoc f' {:args (map-strict first not-atoms)
                                                  :body (peval2' body (into {} atoms))})))
                      ;; Maybe aggressive inlining isn't correct?
                      (let [{:keys [args body]} (get @fdef f')]
                        (peval2' body (into {} (map-strict vector args (map-strict second not-atoms))))))))
                (throw (ex-info "not found function" {:expr ?expr :env env})))
              nil nil
              
              (m/pred atom? ?x) ?x
              (m/pred vector? ?x) (mapv #(peval2' % env) ?x)
              (m/pred seq? ?x) (map-strict #(peval2' % env) ?x)
              (m/pred map? ?x) (reduce-kv (fn [acc k v] (assoc acc (peval2' k env) (peval2' v env))) {} ?x)
              ?x (throw (ex-info "not found expression" {:expr ?x :env env})))]
      (let [result  (peval2' (peval2' expr env) env) ]
        [result @fdef]))))






(defn inline-all [expr fns]
  ;; Need to handle recursion and not infinite loop
  (m/match expr
    (?f & ?args :as ?expr)
    (if-let [f-def (get fns ?f)]
      (inline-all (first
                   ;; Need a better way instead of eval to replace vars
                    (peval2 (:body f-def) (into {} (map vector (:args f-def) ?args )) fns))
                  fns)
      (map-strict #(inline-all % fns) ?expr))
    ?x ?x))


(peval2 1 {} {})
(peval2 "asd" {} {})
(peval2 `(+ 2 2) {} {})
(peval2 '(if x (clojure.core/+ (clojure.core/+ 2 12 23 32412) y 4)) {'x true} {})
(peval2 '(do-things 1 y) {} {'do-things {:args '[x y]
                                         :body `(+ ~'x ~'y (clojure.core/+ 12 32 21))}})


(def example (peval2 '(let [x 3] (exp n x))
                     {}
                     {'exp {:args '[x n]
                            :body '(if (clojure.core/= n 0)
                                     1
                                     (clojure.core/* x (exp x (clojure.core/- n 1))))}}))


(def insert
  '(if (clojure.core/empty? coll)
    [[k v]]
    (clojure.core/conj coll [k v])))


(def lookup
  '(if (clojure.core/empty? coll)
     nil
     (if (clojure.core/= (clojure.core/ffirst coll) x)
       (clojure.core/second (clojure.core/first coll))
       (lookup (clojure.core/rest coll) x))))



;; What if had a stack of what sort of node were in? So I could know
;; that this is a vector context and check that.


(def interpreter
  '(let [tag (clojure.core/get ast :tag)]

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


(defn create-specializer* [pattern]
  (let [arg (gensym "arg")]
    `(fn [~arg]
       ~(clojure.walk/postwalk (fn [x] (if (logic-variable? x)
                                         `(quote ~x)
                                         x))
                               (first
                                (peval2 `(~'interpret ~(zyntax/parse pattern) ~arg [] ~(list))
                                        {}
                                        {'insert {:args '[coll k v]
                                                  :body insert}
                                         'lookup {:args '[coll x]
                                                  :body lookup}
                                         'interpret {:args '[ast form smap]
                                                     :body interpreter}}))))))



(zyntax/parse '(?x [?y ?q] ?z))

(create-specializer* '[?x ?y ?z])


(defmacro specializer [pattern]
  (create-specializer* pattern))


(def partial-evaled (specializer [?x ?y ?z]))

(def epsilon (fn [x] (m/match x
                       [?x ?y ?z]
                       [?x ?y ?z])))

(time 
 (dotimes [_ 10000]
   (partial-evaled [1 2 3])))

(time 
 (dotimes [_ 10000]
   (epsilon [1 2 3])))








