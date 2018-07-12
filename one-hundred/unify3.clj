(ns unify3)

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

(def unify)
(def unify-terms)

(defn unify [env x y]
  (let [x-val (lookup env x)
        y-val (lookup env y)]
    (cond 
      (variable? x-val) (add-var env x-val y-val)
      (variable? y-val) (add-var env y-val x-val)
      :else (unify-terms env x-val y-val))))

(defmulti unify-terms (fn [env x y] [(type x) (type y)]))

(defmethod unify-terms :default [env x y]
  (if (= x y) 
    env
    :unify/failed))

(defn reducer [env [x y]]
  (unify env x y))

(defmethod unify-terms 
  [clojure.lang.Sequential clojure.lang.Sequential] 
  [env x y]
  (reduce reducer env (map vector x 
                           (concat y (repeat :unify/failed)))))

(defn failed? [x]
  (= x :unify/failed))

(defn match-single [value [pattern consequence]]
  [(unify {} pattern value) consequence])

(defn first-match [value patterns]
  (->> patterns
       (map (partial match-single value))
       (drop-while (comp failed? first))
       first))

(defn substitute-all [env vars]
  (clojure.walk/postwalk (partial lookup env) vars))

(defn match* [value & patterns]
  (let [[env consequence] (first-match value (partition 2 patterns))]
    (substitute-all env consequence)))

(defmacro match [value & patterns] 
  `(match* ~value ~@(map (fn [x] `(quote ~x)) patterns)))

(defmacro defmatch [name & patterns]
  `(defn ~name [value#]
     (clojure.core/eval (match* value# ~@(map (fn [x] `(quote ~x)) patterns)))))

(defmatch eval'
  (fn [x] body) (fn [x] body))

(ns-unmap *ns* 'eval)


(eval' '(fn [x] x))
