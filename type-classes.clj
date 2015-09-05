;; Anything you type in here will be executed
;; immediately with the results shown on the
;; right.


(defn register-type [lookup t f]
  (swap! lookup #(assoc % t f)))


(defn create-type [name obj]
  (with-meta obj
    (merge (meta obj) {:type name})))


(defn create-method-body [proto [name args]]
  (let [proto-name (symbol (.toLowerCase proto))
        lookup-name (symbol (str name "-lookup"))]
    `(do
       (def ~lookup-name (atom {}))
       (defn ~name ~args
         (((type ~proto-name) @~lookup-name) ~@args)))))


(defmacro defproto [name & methods]
  (let [method-bodies (map (partial create-method-body (str name)) (partition 2 methods))]
    `(do ~@method-bodies)))


(defn implement-method [t [name f]]
  (let [lookup-name (symbol (str name "-lookup"))]
    `(register-type ~lookup-name ~t ~f)))


(defmacro defimpl [proto t & methods]
  (let [method-bodies (map (partial implement-method (keyword t)) (partition 2 methods))]
    `(do ~@method-bodies)))

(defn create-type-constructor [t [name args]]
  `(defn ~name
     ~args
     (with-meta [~(keyword name) ~@args] {:type ~t})))

(defmacro defdata [t & constructors]
  (let [constructor-bodies (map (partial create-type-constructor (keyword t)) (partition 2 constructors))]
    `(do ~@constructor-bodies)))

(defdata Maybe
  Some [x]
  Nothing [])


(defproto Mappable
  fmap [f mappable])


(defimpl Mappable Maybe
  fmap (fn [f obj]
          (if (= (first obj) :Some)
            (Some (f (second obj)))
            (Nothing))))

(defimpl Mappable List
  fmap map)

(def map-anything-plus-2 (partial fmap (partial + 2)))

(Some 2)


(map-anything-plus-2 (Some 2))
(map-anything-plus-2 (create-type :List '(1 2)))






