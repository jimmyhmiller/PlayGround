(ns wander.core11)

[:repeat {:repeat-type :all :coll-type :vector}
 [:mvr {:name '!xs}]]

(def coll [1 2 3])
(def coll (list 1 2 3))

(if (vector? coll)
  (reduce (fn [!xs x]
            (conj !xs x))
          []
          coll)
  :fail)


[:repeat {:repeat-type :all :coll-type :vector}
 [:pred {:value number?}
  [:mvr {:name '!xs}]]]



(if (vector? coll)
  (reduce (fn [!xs x]
            (if (number? x)
              (conj !xs x)
              (reduced :fail)))
          []
          coll)
  :fail)


(defn dispatch [[tag & _]]
  tag)

(defmulti compile #'dispatch)

(def coll-type-to-pred
  {:vector `vector?})

(def coll-type-to-reduce-init
  {:vector []})

(defmethod compile :repeat [[_ {:keys [repeat-type coll-type]} body]]
  `(if (~(coll-type-to-pred coll-type) ~'target)
    (reduce (fn [~'acc ~'target] ~(compile body)) ~(coll-type-to-reduce-init coll-type) ~'target)
    :fail))

(defmethod compile :constant [[_ {:keys [value]}]]
  `(if (= ~'target ~value)
     true
     :fail))

(defmethod compile :pred [[_ {:keys [value]} body]]
  `(if (~value ~'target )
     ~(compile body)
     :fail))

(defmethod compile :mvr-init [[_ {:keys [name]} body]]
  `(let [~name (transient [])]
     ~(compile body)))

(defmethod compile :mvr-append [[_ {:keys [name]} body]]
  `(conj! ~name ~'target))

(time
 (dotimes [x 1000]
   (filterv number? target)))

(time
 (dotimes [x 1000]
   (clojure.core/let
       [!xs (clojure.core/transient [])
        result (if (clojure.core/vector? target)
                 (clojure.core/reduce
                  (clojure.core/fn
                    [acc target]
                    (if (clojure.core/number? target)
                      (clojure.core/conj! acc  target)
                      :fail))
                  !xs
                  target)
                 :fail)]
     (if (not= result :fail)
       (persistent! !xs)
       :fail))))



(compile 
 [:repeat {:repeat-type :all :coll-type :vector}
  [:constant {:value 1}]])


(def target (into [] (range 1000)))

(compile 
 [:repeat {:repeat-type :all :coll-type :vector}
  [:pred {:value `number?}]])



(compile 
 [:mvr-init {:name '!xs}
  [:repeat {:repeat-type :all :coll-type :vector}
   [:mvr-append {:name '!xs}]]])

(compile
 [:mvr-init {:name '!xs}
  [:repeat {:repeat-type :all :coll-type :vector}
   [:pred {:value `number?}
    [:mvr-append {:name '!xs}]]]])




;; data MeanderExpr
;;     = Literal Atom
;;     | RepeatStar Kind Pattern
;;     | RepeatPlus Kind Int Pattern
;;     | Vector [Pattern]
;;     | Seq [Pattern]
;;     | Map [(Pattern, Pattern)]
;;     | Set [Pattern]
;;     | Wildcard
;;     | Or Pattern Pattern
;;     | And Pattern Pattern
;;     | Pred (a -> Bool) Pattern
;;     | Apply (a -> b) Pattern
;;     | Not Pattern
;;     | With [(Name, Pattern)] Pattern
;;     | Unquote RawExpr
