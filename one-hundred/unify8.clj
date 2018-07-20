(ns unify
  (:import [clojure.lang Sequential]))

(defn variable? [x]
  (symbol? x))

(defn add-var [env var val]
  (assoc env var val))

(defn lookup [env var]
  (if-let [val (get env var)]
    (if (variable? val)
      (lookup env val)
      val)
    var))

(defn failed? [x]
  (= x :unify/failed))

(def unify)
(def unify-terms)

(defn unify [env x y]
  (if (map? env)
    (let [x-val (lookup env x)
          y-val (lookup env y)]
      (cond 
        (variable? x-val) (add-var env x-val y-val)
        (variable? y-val) (add-var env y-val x-val)
        :else (unify-terms env x-val y-val)))
    env))

(defmulti unify-terms
  (fn [env x y] [(type x) (type y)]))

(defn reducer [env [x y]]
  (unify env x y))

(defmethod unify-terms [Sequential Sequential]
  [env x y]
  (if (= (count x) (count y))
    (reduce reducer env (map vector x y))
    :unify/failed))

(defmethod unify-terms :default [env x y]
  (if (= x y)
    env
    :unify/failed))





(defn matcher [value]
  (fn [pattern]
    (unify {} value pattern)))

(defn first-match [value patterns]
  (->> (partition 2 patterns)
       (map (juxt (comp (matcher value) first) second))
       (filter (comp (complement failed?) first))
       (first)))

(defn substitute [[env expr]]
  (clojure.walk/postwalk (partial lookup env) expr))

(defn match* [value patterns]
  (substitute (first-match value patterns)))

(defmacro match [value & patterns]
  `(eval (match* ~value (quote ~patterns))))

(defmacro defmatch [name & patterns]
  `(defn ~name [& args#]
     (match args# ~@patterns)))






(defn match-clause [clause facts env]
  (->> facts
       (map (partial unify env clause))
       (filter (complement failed?))))

(defn match-all [clause facts envs]
  (mapcat (partial match-clause clause facts) envs))

(defn process-query [clauses facts envs]
  (if (empty? clauses)
    envs
    (recur (rest clauses)
           facts
           (match-all (first clauses)
                      facts
                      envs))))

(defn q* [{:keys [find where]} db]
  (let [facts (process-query where db [{}])]
    (map substitute (map vector facts (repeat find)))))


(defmacro q [query db]
  `(q* (quote ~query) db))


(comment

  (def db
    [[1 :age 26]
     [1 :name "jimmy"]
     [2 :age 26]
     [2 :name "steve"]
     [3 :age 24]
     [3 :name "bob"]
     [4 :address 1]
     [4 :address-line-1 "123 street st"]
     [4 :city "Indianapolis"]])



  (q {:find {:name name}
      :where [[_ :name name]]}
     db)


  (q {:find {:name name
             :age age}
      :where [[e :name name]
              [e :age age]]}
     db)




  (q {:find {:name1 name1
             :name2 name2}
      :where [[e1 :name name1]
              [e2 :name name2]
              [e1 :age age]
              [e2 :age age]]}
     db)



  (q {:find {:name name
             :address-line-1 address-line-1
             :city city}
      :where [[e :name name]
              [a :address e]
              [a :address-line-1 address-line-1]
              [a :city city]]}
     db)
  )
