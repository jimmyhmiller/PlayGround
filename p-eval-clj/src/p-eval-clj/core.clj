(ns p-eval-clj.core
  (:require [meander.epsilon :as m]
            [clojure.walk :as walk]
            [meander.syntax.zeta :as zyntax]
            [clojure.string :as string]))


(defn logic-variable? [x]
  (and (symbol? x)
       (string/starts-with? (name x) "?")))

(defn atom? [expr]
  (if (coll? expr)
    ;; this isn't right
    (every? (some-fn atom? logic-variable?) expr)
    (boolean
     ((some-fn number? string? fn? keyword? boolean? nil?) expr))))

(defn eval' [expr env]
  (m/match expr
    (m/pred atom? ?x) ?x

    (m/and (m/pred symbol? ?x))
    (if-let [val (get env ?x)]
      val
      (throw (ex-info "error" {:x ?x :env env})))

    ((m/and (m/symbol "clojure.core" ?sym) ?f) & ?args)
    (apply (resolve ?f) (map #(eval' % env) ?args))

    (if ?pred ?t ?f)
    (if (eval' ?pred env)
      (eval' ?t env)
      (eval' ?f env))

    (?f & ?args)
    (let [{:keys [args body]} (get env ?f)]
      (eval' body
             (merge env (into {} (map vector
                                      args
                                      (map #(eval' % env) ?args))))))
    nil nil
    ?x (throw (ex-info "not found" {:expr ?x :env env}))))




(eval' 1 {})
(eval' "asd" {})
(eval' `(+ 2 2) {})
(eval' '(if x true false) {'x true})
(eval' '(do-things 1 2) {'do-things {:args '[x y]
                                     :body `(+ ~'x ~'y)}})



(defn peval' [expr env]

  (m/match expr
    (m/pred atom? ?x) ?x

    (m/and (m/pred symbol? ?x))
    (get env ?x ?x)

    ((m/and (m/symbol "clojure.core" ?sym) ?f) & ?args :as ?expr)
    (let [args (map #(peval' % env) ?args)]
      (if (every? atom? args)
        (apply (resolve ?f) args)
        (cons ?f args)))

    (if ?pred ?t)
    (peval' (list 'if ?pred ?t nil) env)

    (if ?pred ?t ?f)
    (let [evaled-pred (peval' ?pred env)]
      (if (atom? evaled-pred)
        (if evaled-pred
          (peval' ?t env)
          (peval' ?f env))
        `(if ~evaled-pred
           ~(peval' ?t env)
           ~(peval' ?f env))))

    (?f & ?args)
    (let [{:keys [args body]} (get env ?f)]
      (peval' body
              (merge env (into {} (map vector
                                       args
                                       (map #(peval' % env) ?args))))))
    nil nil
    ?x (throw (ex-info "not found" {:expr ?x :env env}))))

(peval'
 (walk/macroexpand-all
  `(cond
     (= ~'op :add) +
     (= ~'op :minus) -
     :else :unknown))
 {'op :add})





(peval' 1 {})
(peval' "asd" {})
(peval' `(+ 2 2) {})
(peval' '(if x (clojure.core/+ (clojure.core/+ 2 12 23 32412) y 4)) {'x true})
(peval' '(do-things 1 y) {'do-things {:args '[x y]
                                      :body `(+ ~'x ~'y (clojure.core/+ 12 32 21))}})


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
              (m/match expr

                (m/pred logic-variable?)
                `(quote ~expr)

                (m/and (m/pred symbol? ?x))
                (get env ?x ?x)

                (clojure.core/list & ?args)
                `(clojure.core/list ~@(map-strict #(peval2' % env) ?args))


               (clojure.core/mapcat (fn [?arg] ?body) (clojure.core/list ?x))
               (peval2' ?body (assoc env ?arg ?x))


                ((m/and (m/symbol "clojure.core" ?sym) ?f) & ?args :as ?expr)
                (let [args (map-strict #(peval2' % env) ?args)]
                  (if (every? atom? args)
                    (apply (resolve ?f) args)
                    (cons ?f args)))

                (let [?var ?val] ?body)
                (let [evaled-val (peval2' ?val env)]
                  (if (atom? evaled-val)
                    (peval2' ?body (assoc env ?var evaled-val))
                    `(~'let [~?var ~evaled-val]
                       ~(peval2' ?body env))))

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
                        (let [{:keys [args body]} (get @fdef f')
                              renamed-args (map-strict gensym args)
                              renamed-body (peval2' body (into {} (map-strict vector args renamed-args)))]
                          (peval2' renamed-body (into {} (map-strict vector renamed-args
                                                                     (map-strict second not-atoms))))))))
                  (throw (ex-info "not found function" {:expr ?expr :env env})))
                nil nil
               
                (m/pred atom? ?x) ?x
                (m/pred vector? ?x) (mapv #(peval2' % env) ?x)
                ?x (throw (ex-info "not found expression" {:expr ?x :env env}))))]
      (let [result (peval2' expr env)]
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



(def interpreter
  '(let [tag (clojure.core/get ast :tag)]
     (if (clojure.core/= tag :literal)
       (let [x (clojure.core/get ast :form)]
         (if (clojure.core/= x form)
           (clojure.core/list smap)))
       (if (clojure.core/= tag :logic-variable)
         (let [other-form (clojure.core/get smap ast)]
           (if other-form
             (if (clojure.core/= other-form form)
               (clojure.core/list smap))
             (clojure.core/list (clojure.core/assoc smap ast form))))
         (if (clojure.core/= tag :cat)
           (if (clojure.core/seq form)
             (clojure.core/mapcat
              (fn [smap']
                (interpret (clojure.core/get ast :next) (clojure.core/next form) smap'))
              (interpret (clojure.core/get ast :pattern) (clojure.core/nth form 0) smap)))
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
                                (peval2 `(~'interpret ~(zyntax/parse pattern) ~arg {})
                                        {}
                                        {'interpret {:args '[ast form smap]
                                                     :body interpreter}}))))))

(add-watch #'peval2 :thing (fn [_ _ _ _]
                             (clojure.pprint/pprint (create-specializer* '[?x ?y ?z]))))



(create-specializer* '[?x ?y ?z])

(peval2 '?v {} {})
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








