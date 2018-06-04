(ns unify-oop.core
  (:require [clojure.string :as string]
            [clojure.core.match :refer [match]]))

(defn variable [name index] 
  [:var {:name name :index index}])

(defn const [val] 
  [:const {:val val}])

(defn absent []
  [:absent {}])

(defn present [const]
  [:present {:const const}])

(defn arrow [left right]
  [:arrow {:left left :right right}])

(defn record [labels]
  [:record {:labels labels}])

(defn extension [labels row]
  [:extension {:labels labels :row row}])

(defn empty []
  [:empty {}])

(defn t-var [name]
  (variable name 0))

(def gen-num 
  (let [count (atom 0)]
    (fn []
      (swap! count inc))))

(defn gen-var [name]
  (let [num (gen-num)]
    (t-var (str name num))))


(defn gen-logic-var [name]
  (if name
    (symbol (str "?" name (gen-num)))
    nil))

(defn logic-variable? [x]
  (and (symbol? x)
       (string/starts-with? (name x) "?")))

(defn find [var var-map]
  (if-let [val (get var-map var)]
    (if (logic-variable? val)
      (walk-var-binding val var-map)
      val)
    var))

(defn add-equivalence [var val var-map]
  (when var-map
    (assoc var-map var val)))

(defn equiv [var-map [x y]]
  (if (map? var-map)
    (let [x' (walk-var-binding x var-map)
          y' (walk-var-binding y var-map)]
      (cond 
        (= x' y') var-map 
        (logic-variable? x') (add-equivalence x' y' var-map)
        (logic-variable? y') (add-equivalence y' x' var-map)
        :else :unify/failed))
    var-map))

(defn lookup [x env succ fail]
  (if (contains? env x)
    (succ (second (env x)))
    (fail)))

(defn add-var [env var node]
  (assoc env var node))

(defn node [expr name]
  (let [[type info] expr]
    [type (assoc info :logic-var (gen-logic-var name))]))


(defn translate [expr env]
  (match [expr]
         [[:var {:name name}]] (lookup 
                                expr 
                                env 
                                (fn [x] [x env])
                                (fn [] (let [n (node expr name)]
                                         [n (add-var env expr n)])))
         [[:const info]] [(node expr nil) env]
         [[:absent]] [(node expr "absent") env]
         [[:present {:const const}]] (let [[n env1] (translate const env)]
                                       [(node (present n) "present") env1])
         [[:arrow {:left left
                   :right right}]] (let [[left1 env1] (translate left env)
                                         [right1 env2] (translate right env1)]
                                     [(node (arrow left1 right1) "arrow") env2])
         [[:record {:labels labels}]] (let [[n env1] (translate labels env)]
                                        [(node n "record") env1])))




(defn sym-from-keyword [k]
  (symbol (namespace k) (name k)))


