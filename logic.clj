(defn statement? [statement]
  (or (= (type statement) clojure.lang.PersistentVector)
      (seq? statement)))

(defn conj? [statement]
  ((comp not nil?) (some #{:and} statement)))

(defn alt? [statement]
  ((comp not nil?) (some #{:or} statement)))

(defn if-continue [x pred f] ; The params with be excuted, but purity makes that not matter
  (if (pred x)
    (f x)
    x))

(defn st-fn [statement f]
  (if-continue statement statement? f))

(defn conj-fn [statement f]
  (st-fn statement
         (fn [statement]
           (if-continue statement conj? f))))

(defn alt-fn [statement f]
  (st-fn statement
         (fn [statement]
           (if-continue statement alt? f))))


(defmacro defstatementfn [fn-name params & body]
  `(defn ~fn-name ~params
     (st-fn ~(first params)
            (fn [~(first params)] ~@body))))

(defmacro defaltfn [fn-name params & body]
  `(defn ~fn-name ~params
     (alt-fn ~(first params)
            (fn [~(first params)] ~@body))))

(defmacro defconjfn [fn-name params & body]
  `(defn ~fn-name ~params
     (conj-fn ~(first params)
            (fn [~(first params)] ~@body))))


(defconjfn remove-t-conj [statement]
  (into [] (remove #(= :T %) statement)))

(defaltfn remove-f-alt [statement]
  (into [] (remove #(= :F %) statement)))

(defconjfn reduce-f-conj [statement]
  (if (some #{:F} statement) :F statement))

(defaltfn reduce-t-alt [statement]
  (if (some #{:T} statement) :T statement))

(defstatementfn reduce-1-pred [statement]
  (if (<= (count statement) 2)
    (second statement)
    statement))

(defstatementfn remove-duplicates [statement]
  (into [] (distinct statement)))

(defstatementfn remove-nil [statement]
  (into [] (remove #(or (nil? %) (and (statement? %) (empty? %))) statement)))

(defstatementfn replace-var [statement var value]
  (into [] (replace {var value} statement)))

(defn reduce-statement [statement]
  (-> statement
      (remove-duplicates)
      (remove-t-conj)
      (remove-f-alt)
      (reduce-f-conj)
      (reduce-t-alt)
      (remove-nil)
      (reduce-1-pred)))

(defn recursive-reduce [statement]
  (let [statement (reduce-statement statement)]
    (cond (not (statement? statement))
          (reduce-statement statement)
          (not-any? statement? statement)
          (reduce-statement statement)
          :else (reduce-statement (mapv recursive-reduce statement)))))


(reduce-statement [:or :p :q [:or :p :q]])

(recursive-reduce  [:or :F [:or :F [:or :F [:or :F [:and :p]] [:or :F :F]]]])


(recursive-reduce [:and :p [:or
                            [:or :F [:or :F [:or :F [:or :F [:and :p]] [:or :F :F]]]]
                            [:or :F [:or :F [:or :F [:or :F [:and :p]] [:or :F :F]]]]
                            [:and :p :T :x] :y]])


(reduce-statement [:or :p :q :T])
(reduce-statement [:or :p :q :F])


(reduce-statement [:and :p :T])
(reduce-statement [:and :p :F])
(reduce-statement [:or :F :F])


