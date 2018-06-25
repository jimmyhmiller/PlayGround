(ns unify2)


(defn var? [x]
  (symbol? x))

(defn lookup [env var] 
  (if-let [val (get env var)]
    (if (var? val)
      (lookup env val)
      val)
    var))

(defn add-var [env var val]
  (assoc env var val))

(defn unify [env x y] 
  (let [val-x (lookup env x)
        val-y (lookup env y)]
    (cond
      (var? val-x) (add-var env val-x val-y)
      (var? val-y) (add-var env val-y val-x)
      :else (unify-terms env val-x val-y))))

(defmulti unify-terms (fn [env x y] [(type x) (type y)]))

(defn reducer [env [x y]]
  (if (= y :unify/failed)
    :unify/failed
    (unify env x y)))

(defmethod unify-terms [clojure.lang.Sequential clojure.lang.Sequential] [env x y] 
  (reduce reducer env (map vector x (concat y (repeat :unify/failed)))))

(defmethod unify-terms :default [env x y]
  (if (= x y)
    env
    :unify/failed))

(defn substitute-all [env vars] 
  (if (map? env)
    (clojure.walk/postwalk (partial lookup env) vars)
    env))

(defn failed? [val]
  (= val :unify/failed))

(defn match-first [value patterns]
  (->> patterns
       (map (fn [[pattern con]] [(unify {} pattern value) con]))
       (drop-while (comp failed? first))
       first))

(defn match* [value & patterns]
  (let [[env consequence] (match-first value (partition 2 patterns))]
    (substitute-all env consequence)))

(defmacro match [value & patterns]
  `(match* ~value ~@(map (fn [x] `(quote ~x)) patterns)))

(defmacro defmatch [name & patterns]
  `(defn ~name [value#]
     (match value# ~@patterns)))

