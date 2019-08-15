(ns wander.core9
  (:require [meander.epsilon :as m]
            [meander.strategy.epsilon :as r]
            [clojure.set :as set]
            [clojure.walk :as walk]
            [clojure.string :as string]))



(defn lvar? [x]
  (and (symbol? x) (string/starts-with? (name x) "?")))

(defn all-containing-lvars [data]
  (set (filter lvar? (flatten (seq data)))))

(defn has-any-lvars [lvars]
  (if (coll? lvars)
    (boolean (some? (set (filter lvar? (flatten (seq lvars))))))
    false))

(defn has-some-lvars [source-vars lvars var]
  (if (coll? (lvars var))
    (some (all-containing-lvars (lvars var)) (keys source-vars))
    false))

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
  ;; In the single instance case we want to remove things that are probably
  ;; just data, strings and numbers are good examples of this.
  (let [other-preds (if (= (count example-vars) 1) 
                      (comp #(remove number? %) #(remove string? %)) identity)
        values (map (comp set
                          other-preds
                          #(remove lvar? %)
                          #(remove has-any-lvars %)
                          #(tree-seq coll? seq %) :vars) example-vars)]
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

(defn first-constants [lvars const]
  (first (filter (comp #{const} second) (:vars lvars))))

(defn fill-in-lvars [lvars1 lvars2 constants]
  (let [constants (seq constants)
        replacements1 (map first (map (partial first-constants lvars1) constants))
        replacements2 (map first (map (partial first-constants lvars2) constants))]
    {:input (reduce (fn [acc x] (update-in acc [:vars] #(dissoc % x))) 
             (reduce replace-vars lvars1 (map vector replacements1 replacements2))
             replacements2)
     :output (reduce (fn [acc x] (update-in acc [:vars] #(dissoc % x))) lvars2 replacements2)}))


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


(defn find-lvar-replacements [vars]
  ;; Need to handle duplicates 
  (->> vars
       (tree-seq coll? seq)
       (filter map-entry?)
       (filter (comp keyword? key))
       (filter (comp lvar? val))
       (map (fn [[k v]] [v (symbol (str "?" (name k)))]))
       (into {})))

(defn pretty-lvars [{:keys [input output]}]
  ;; infinite loops on duplicates
  (let [replacements (find-lvar-replacements (:vars input))]
    {:input (walk/postwalk-replace replacements input)
     :output (walk/postwalk-replace replacements output)}))

(defn construct-match [{:keys [input output]}]
  `(~'r/rewrite
    ~(construct-clause (:vars input) input)
    ~(construct-clause (:vars output) output)))

(defn infer-rewrite* [input-outputs]
  (let [pairs (partition 2 input-outputs)
        inputs (map assign-lvars (map first pairs))
        outputs (map assign-lvars (map second pairs))
        representative-input (fill-in-constants (first inputs)
                                                (find-constants inputs))
        representative-output (fill-in-constants (first outputs)
                                                 (find-constants inputs))
        representative-output' (fill-in-constants representative-output
                                                  (find-constants outputs))
        input-output (pretty-lvars
                      (fill-in-lvars representative-input
                                     representative-output'
                                     (find-constants [representative-input representative-output'])))]
    `~(construct-match input-output)))

(defn infer-rewrite [& input-outputs]
  (infer-match* input-outputs))

(defmacro infer-transform [& input-outputs]
  (infer-rewrite* input-outputs))



(println "\n\n\n\n")

(def transformation
  (infer-transform
   {:name "jimmy"
    :age 27}

   {:info
    {:name "jimmy"
     :age 27}}))

(transformation 
 {:name "thing" :age 53})


(infer-rewrite
 {:name "jimmy"
  :age 27}

 {:info 
  {:name "jimmy"
   :age 27}})




(infer-rewrite

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

(infer-rewrite [1 2 3]
             [3 2 1])



(infer-rewrite
 [1 2 3 4 5 6 7 124 123]
 {:thing 1
  :stuff 2
  :nested {:thing-here [4 5 6 7]}
  :more [124 123]})

(def transform
  (infer-transform
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
                :zipcode "46203"}}))


(transform 
 {:name "Jimmy Miller"
  :address
  {:address1 "432 Thing"
   :address2 "apt 20"
   :city "Statesburg"
   :state "KY"
   :zip "231512"}})

(infer-rewrite
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
              :zipcode "46203"}})


(infer-rewrite
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










