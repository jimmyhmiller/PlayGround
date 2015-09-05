;; Anything you type in here will be executed
;; immediately with the results shown on the
;; right.


(def database
  {:person
   [[:janice]
    [:jimmy]
    [:archy]
    [:julie]
    [:fred]
    [:cindy]
    [:herby]
    [:ginny]]
   :parent
   [[:jimmy :fred]
    [:jimmy :cindy]
    [:janice :archy]
    [:janice :julie]
    [:cindy :herby]
    [:cindy :ginny]]})


(def rules {:grand [[:?x :?z] [[:parent :?x :?y] [:parent :?y :?z]]]})


(defn var? [word]
  (.contains (str word) "?"))

(defn replace-recursive [replacement rule]
  (cond
   (sequential? rule) (map (partial replace-recursive replacement) rule)
   (= (first replacement) rule) (second replacement)
   :else rule))


(defn match [pattern fact]
  (cond
   (empty? pattern) true
   (or
    (var? (first pattern))
    (= (first pattern) (first fact))) (match (rest pattern) (rest fact))
   :else false))



(defn get-facts-for-query [[coll & query]]
  (coll database))



(let [rule [:parent :?x :?z]
      query [:parent :jimmy :?z]
      query-body (rest query)]
  (->> query
       (get-facts-for-query)
       (filter (partial match query-body))
       (map (fn [r] (replacements query-body r)))))


  (get-facts-for-query [:parent :?x :?z])


(defn replacements [rule query-body]
  (filter #(not= nil %) (map (fn [r q]
         (if (and (var? r) (not (var? q)))
           [r q]
           nil)) rule query-body)))







