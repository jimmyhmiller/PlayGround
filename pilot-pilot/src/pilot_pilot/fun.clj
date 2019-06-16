(ns pilot-pilot.fun)


(defmacro rules [& args]
  `(quote ~args))




(defn simple-replacement [left right]
  `(fn [x#]
     (if (= x# ~left)
       ~right
       x#)))



(defn substitute [var expr]
  `(fn [val#]
     (clojure.walk/postwalk (fn [x#] 
                              (println x# (quote ~var) (= x# (quote ~var)) val#)
                              (if (= x# (quote ~var)) val# x#)) ~expr)))

(defmacro rewrite [{:keys [left right]}]
  (let [has-vars?  (= 1 (bounded-count 1 (filter symbol? (if (coll? left) left [left]))))]
    (cond (not has-vars?)
          (simple-replacement left right)
          (symbol? left) (substitute left right))))




((substitute 'x 3)
 '(x))

(macroexpand
 (quote
  (rewrite {:left x
            :right '(x 2)})))


((rewrite {:left x
           :right '(x 2)})

 5)

((rewrite
  {:left x
   :right :thing})
 :thin)

(rules
 (0 + x => x)
 (x + 0 => x)
 (x + y => (clojure.core/+ x y))
)
