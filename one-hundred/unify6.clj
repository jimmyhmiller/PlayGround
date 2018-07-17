(ns unify
  (:import [clojure.lang Sequential]))

(defn variable? [x]
  (symbol? x))

(defn add-var [env var val]
  (assoc env var val))

(defn lookup [env var]
  (if-let [val (get env var)]
    (if (var? val)
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
  (->> patterns
       (map (juxt 
             (comp (matcher value) first) 
             second))
       (filter (comp (complement failed?) first))
       first))

(defn substitute [[env expr]]
  (clojure.walk/postwalk (partial lookup env) expr))

(defn match* [value patterns]
  (->> patterns
       (partition 2)
       (first-match value)
       substitute))

(defmacro match [value & patterns]
  `(eval (match* ~value (quote ~patterns))))

(defmacro defmatch [name & patterns]
  `(defn ~name [& args#]
     (match args# ~@patterns)))

(defmatch fib 
  [0] 0
  [1] 1
  [n] (+ (fib (- n 1))
         (fib (- n 2))))


(defn clause-matcher [facts query] 
  (fn [env]
    (->> facts
         (map (partial unify env query))
         (filter (complement failed?)))))

(defn all-matches [facts clause envs]
  (mapcat (clause-matcher facts clause) envs))

(defn process-query [facts clauses envs]
  (if (empty? clauses)
    envs
    (recur facts
           (rest clauses)
           (all-matches facts (first clauses) envs))))

(defn q* [{:keys [find where]} db]
  (let [results (process-query db where [{}])]
    (map #(substitute [% find]) results)))


(defmacro q [query db]
  `(q* (quote ~query) ~db))


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
    :where [[e :name name]]} db)
