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
  ([x] (= x :unify/failed))
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
  (if (coll? pattern)
    (substitute-all (unify-all {} pattern value) consequece)
    (substitute-all (unify {} pattern value) consequece)))



(defn match** [value & patcons]
  (drop-while (partial = :unify/failed)
              (map (fn [[pattern consequence]] 
                     (substitute-all (unify-all {} pattern value) consequence))
                   (partition 2 patcons))))


                                       
(defn match* [value pat con & patcons]
  (if (empty? patcons) (match-1 value pat con)
      (let [potential-match (match-1 value pat con)]
        (if (failed? potential-match)
          (apply match* (cons value patcons))
          potential-match))))

(defmacro match [value & args]
  `(match* ~value ~@(map (fn [x] `(quote ~x)) args)))




(match '(fn [x] x)
       ('fn arg body) [arg body])

(-> {}
    (unify 'x 'y)
    (unify 'y 'z)
    (unify 'z 2)
    (lookup 'x))



