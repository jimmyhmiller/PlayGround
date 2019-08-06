(ns wander.core9
  (:require [meander.epsilon :as m]
            [meander.strategy.epsilon :as r]
            [clojure.set :as set]
            [clojure.walk :as walk]
            [clojure.string :as string]))




(defn assign-lvars [data]
  (let [vars (atom [])
        new-structure 
        ((r/bottom-up
          (r/rewrite
           ?x ~(let [x (gensym "?x_")] 
                 (swap! vars conj [x ?x])
                 x)))
         data)]
    {:vars (into {} @vars)
     :data new-structure}))

(defn find-constants [example-vars]
  (let [values (map (comp set #(map second %) :vars) example-vars)]
    (apply set/intersection values)))

(defn replace-vars [data [var val]]
  (->> data
      ((r/n-times 2
         (r/bottom-up
          (r/attempt
           (fn [t]
             ((r/rewrite
               [[?var ?val] ?var] ?val)
              [[var val] t]))))))
      (walk/postwalk (r/attempt (r/rewrite [?val ?val] nil)))))

(defn fill-in-constants [lvars constants]
  (reduce replace-vars lvars (filter (comp constants second) (:vars lvars))))

(def lvars
  (assign-lvars 
   {:name "jimmy"
    :age 27}))
(defn log [x]
  (println x)
  x)


(defn first-constants [lvars const]
  (first (filter (comp #{const} second) (:vars lvars))))

(defn fill-in-lvars [lvars1 lvars2 constants]
  (let [constants (seq constants)
        replacements1 (map first (map (partial first-constants lvars1) constants))
        replacements2  (map first (map (partial first-constants lvars2) constants))]
    (println replacements1 replacements2 constants lvars1 lvars2)
    {:input (reduce (fn [acc x] (update-in acc [:vars] #(dissoc % x))) 
             (reduce replace-vars lvars1 (map vector replacements1 replacements2))
             replacements2)
     :output (reduce (fn [acc x] (update-in acc [:vars] #(dissoc % x))) lvars2 replacements2)}))



(defn lvar? [x]
  (and (symbol? x) (string/starts-with? (name x) "?")))


(defn all-containing-lvars [data]
  (set (filter lvar? (flatten (seq data)))))

(defn has-some-lvars [source-vars lvars var]
  (if (coll? (lvars var))
    (some (all-containing-lvars (lvars var)) (keys source-vars))
    false))



(defn construct-clause [source-vars lvars]
  (let [entry (get (:vars lvars) (:data lvars))]
    ((r/until =
       (fn [t]
         (walk/postwalk
          (r/attempt
           (r/rewrite
            [(m/and (m/not (m/pred lvar?)) ?k)
             (m/and (m/pred lvar? ?v)
                    (m/pred #(contains? source-vars %) ?v))]
            [?k ~(source-vars ?v)]

            [(m/and ?k (m/not (m/pred lvar?)))

             (m/and (m/pred lvar? ?v)
                    (m/pred (partial has-some-lvars source-vars (:vars lvars)) ?v))]


            [?k ~((:vars lvars) ?v)]
            
            [(m/and (m/pred lvar? ?k)
                    (m/pred #(contains? source-vars %) ?k))
             (m/and (m/pred lvar? ?v)
                    (m/pred #(contains? source-vars %) ?v))]
            [~(source-vars ?k) ~(source-vars ?v)]

            [(m/and (m/pred lvar? ?k)
                    (m/pred #(contains? source-vars %) ?v))
             (m/and (m/not (m/pred lvar?)) ?v)]
            [?k ?v]))
          t))) entry)))

(defn construct-match [{:keys [input output]}]
  `(~'r/rewrite
    ~(construct-clause (:vars input) input)
    ~(construct-clause (:vars output) output)))


(defn infer-match [& input-outputs]
  (let [pairs (partition 2 input-outputs)
        inputs (map assign-lvars (map first pairs))
        outputs (map assign-lvars (map second pairs))
        representative-input (fill-in-constants (first inputs)
                                                (find-constants inputs))
        representative-output (fill-in-constants (first outputs)
                                                 (find-constants inputs))
        representative-output' (fill-in-constants representative-output
                                                 (find-constants outputs))
        input-output (fill-in-lvars representative-input
                                    representative-output'
                                    (find-constants [representative-input representative-output']))]
    `~(construct-match input-output)))



(infer-match

 {:name "jimmy"
  :age 27}

 {:info 
  {:name "jimmy"
   :age 27}}

 {:name "stuff"
  :age 36
  :other-stuff 2}

 {:info
  {:name "stuff"
   :age 36}})


(infer-match
 {:name "Jimmy"
  :address
  {:address1 "123 street ave"
   :address2 "apt 2"
   :city "Townville"
   :state "IN"
   :zip "46203"}}


 {:name "Jimmy"
  :address {:line1 "123 street ave"
            :line2 "apt 2"}
  :city-info {:city "Townville"
              :state "IN"
              :zipcode "46203"}}

 {:name "Jimmy2"
  :address
  {:address1 "12 street ave"
   :address2 "apt 23"
   :city "Town23ville"
   :state "I23N"
   :zip "4622303"}}
 
 {:name "Jimmy2"
  :address {:line1 "12 street ave"
            :line2 "apt 23"}
  :city-info {:city "Town23ville"
              :state "I23N"
              :zipcode "4622303"}})




((r/rewrite
  {:name ?x_24502,
   :address
   {:address1 ?x_24505,
    :address2 ?x_24507,
    :city ?x_24511,
    :state ?x_24513,
    :zip ?x_24515}}
  {:name ?x_24502,
   :address {:line1 ?x_24505, :line2 ?x_24507},
   :city-info {:city ?x_24511, :state ?x_24513, :zipcode ?x_24515}}))






