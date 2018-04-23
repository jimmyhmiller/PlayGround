(ns playing-with-ideas
  (:require [clojure.string :as string]))


(defn dispatcher [obj message & args] 
  (apply (message obj)
         (cons (partial dispatcher obj) args)))

(defn class [obj]
  (partial dispatcher obj))

(defn get-methods [obj]
  (obj constantly))

(defn method-names [obj]
  (keys (get-methods obj)))

(defn extend-object [obj extension]
  (let [methods (get-methods obj)]
    (class (merge methods extension))))

(defn inherit [base-class my-class]
  (extend-object base-class my-class))

(def Animal
  (class {:isAnimal (fn [this] true)}))

(def Dog
  (inherit Animal {:speaks (fn [this] "Bark")}))

(def Insert)
(def Union)

(def Empty
  (class {:isEmpty (fn [this] true)
          :contains (fn [this i] false)
          :insert (fn [this i] (Insert this i))
          :union (fn [this s] s)}))



(defn Insert [s n]
  (if (s :contains n)
    s
    (class
     {:isEmpty (fn [this] false)
      :contains (fn [this i] (or (= i n) (s :contains i)))
      :insert (fn [this i] (Insert this i))
      :union (fn [this s] (Union this s))})))

(defn Counter [initial-count]
  (let [count (atom initial-count)]
    (class {:increment (fn [this] (swap! count inc))
            :get-count (fn [this] @count)})))

(def counter (Counter 12))

(def super-counter 
  (extend-object 
   counter
   {:inc-3 (fn [this] 
             (this :increment)
             (this :increment)
             (this :increment))}))

(counter :get-count)

(super-counter :inc-3)


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


(defn logic-variable? [x]
  (and (symbol? x)
       (string/starts-with? (name x) "?")))

(defn walk-var-binding [var var-map]
  (if-let [val (get var-map var)]
    (if (logic-variable? val)
      (walk-var-binding val var-map)
      val)
    var))

(defn add-equivalence [var val var-map]
  (when var-map
    (assoc var-map var val)))

(defn unify [var-map [x y]]
  (if (map? var-map)
    (let [x' (walk-var-binding x var-map)
          y' (walk-var-binding y var-map)]
      (cond 
        (= x' y') var-map 
        (logic-variable? x') (add-equivalence x' y' var-map)
        (logic-variable? y') (add-equivalence y' x' var-map)
        :else :unify/failed))
    var-map))

(defn unify-all [var-map vals vars]
  (reduce unify var-map
          (map vector vals
               (concat vars (repeat :unify/failed)))))


(defn failed? [unified]
  (if (seqable? unified)
    (not (nil? (some #{:unify/failed} unified)))
    (= unified :unify/failed)))

(defn substitute-all [vars var-map]
  (map (fn [var] (walk-var-binding var var-map)) vars))

(defn match-1 [value pattern consequence]
  (let [var-map (unify-all {} value pattern)]
    (if (failed? var-map)
      :unify/failed
      (substitute-all consequence var-map))))

(defn match* [value pat con & patcons]
  (if (empty? patcons)
    (match-1 value pat con)
    (let [potential-match (match-1 value pat con)]
      (if (failed? potential-match)
        (apply match* (cons value patcons))
        potential-match))))

(match* [1 2 1]
        '[?x ?y] '[?x ?y]
        '[?x ?y ?x] '[?x ?y]
        '[?x ?y ?z] '[?x ?y ?z])

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;


(defn match-query [var-map facts q]
  (remove failed? (map #(unify-all var-map % q) facts)))


(defn process-query
  ([qs facts]
   (process-query (rest qs) facts (match-query {} facts (first qs))))
  ([qs facts var-maps]
   (if (empty? qs)
     var-maps
     (process-query
      (rest qs) 
      facts 
      (mapcat (fn [var-map]
                (match-query var-map facts
                             (first qs)))
              var-maps)))))


(defn query [facts select qs]
  (set (map #(substitute-all select %) (process-query qs facts))))


(query
 [[1 :likes :pizza]
  [2 :likes :nachos]
  [1  :things :stuff]
  [2 :things :otherstuff]]

 '[?food ?stuff]
 
 '[[?e :likes ?food]
   [?e :things ?stuff]])

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
