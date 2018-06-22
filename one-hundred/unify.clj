(ns unify)


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

(defn failed? 
  ([env] (not (map? env)))
  ([env x y]
   (or (not (map? env))
       (= x :unify/failed)
       (= y :unify/failed))))

(defn unify [env x y]
  (if (failed? env x y)
    :unify/failed
    (let [val-x (lookup env x)
          val-y (lookup env y)]
      (cond
        (= val-x val-y) env
        (variable? val-x) (add-var env val-x val-y)
        (variable? val-y) (add-var env val-y val-x)
        :else :unify/failed))))

(defn unify-all [env vars vals]
  (reduce (fn [env [var val]] (unify env var val)) 
          env
          (map vector vars (concat vals (repeat :unify/failed)))))


(defn substitute-all [env vars] 
  (if (map? env)
    (clojure.walk/postwalk (fn [var] (lookup env var)) vars)
    env))

(defn match-1 [value pattern consequece]
  (substitute-all (unify-all {} pattern value) consequece))





(-> {}
    (unify 'x 'y)
    (unify 'y 'z)
    (unify 'z 2)
    (lookup 'x))



