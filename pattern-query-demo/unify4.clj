(ns unify4)


(defn var? [x]
  (symbol? x))

(defn add-var [env var val]
  (assoc env var val))

(defn lookup [env var]
  (if-let [val (get env var)]
    (if (var? val)
      (lookup env val)
      val)
    var))

(def unify)
(def unify-terms)

(defn unify [env x y]
  (if (map? env)
    (let [x-val (lookup env x)
          y-val (lookup env y)]
      (cond
        (var? x-val) (add-var env x-val y-val)
        (var? y-val) (add-var env y-val x-val)
        :else (unify-terms env x-val y-val)))
    env))

(defmulti unify-terms 
  (fn [env x y] [(type x) (type y)]))

(defmethod unify-terms :default [env x y]
  (if (= x y)
    env
    :unify/failed))

(defn reducer [env [x y]]
  (unify env x y))

(defmethod unify-terms [clojure.lang.Sequential clojure.lang.Sequential] [env x y]
  (if (= (count x) (count y))
    (reduce reducer env (map vector x y))
    :unify/failed))

(defn failed? [x]
  (= x :unify/failed))

(defn match-single [value pattern]
  (unify {} pattern value))

(defn first-match [value patterns]
  (let [match-value (partial match-single value)]
    (->> patterns
         (map (juxt (comp match-value first) second))
         (drop-while (comp failed? first))
         first)))

(first-match [1 2 3] '[[[x y z] [x y z]]])

(defn substitute-all [env vars]
  (clojure.walk/postwalk (partial lookup env) vars))

(defn match* [value & patterns]
  (let [[env consequence] (first-match value (partition 2 patterns))]
    (substitute-all env consequence)))

(defmacro match [value & patterns]
  `(eval (match* ~value ~@(map (fn [x] `(quote ~x)) patterns))))

(defmacro defmatch [name & patterns]
  `(defn ~name [value#]
     (match value# ~@patterns)))

(defmatch fib
  0 0
  1 1
  n (+ (fib (- n 1))
       (fib (- n 2))))


(defn match-clause [env facts query]
  (->> facts
       (map (partial unify env query))
       (filter (complement failed?))))

(defn process-query
  ([clauses facts]
   (process-query [{}] clauses facts))
  ([envs clauses facts]
   (if (empty? clauses) 
     envs
     (recur
      (mapcat (fn [env] (match-clause env facts (first clauses))) envs)
      (rest clauses)
      facts))))

(def facts
  [[1 :age 26]
   [1 :name "jimmy"]
   [2 :age 26]
   [2 :name "steve"]])

(def query1
  '[[e1 :age age]
    [e2 :age age]])


(process-query query1 facts)
