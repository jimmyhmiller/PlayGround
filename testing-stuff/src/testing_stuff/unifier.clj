(ns testing-stuff.unifier
  (:import (clojure.lang Sequential))
  (:require [clojure.walk]
          [clojure.string :as string]))

(defn variable? [x]
  (and (symbol? x)
       (string/starts-with? (name x) "?")))

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

(defn substitute [[env expr]]
  (clojure.walk/postwalk (fn [x] (lookup env x)) expr))


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
           (match-all (first clauses) facts envs))))

(defn q* [{:keys [find where]} db]
  (let [envs (process-query where db [{}])]
    (map substitute (map vector envs (repeat find)))))

(defmacro q [query db]
  `(q* (quote ~query) db))
