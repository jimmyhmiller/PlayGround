;; Test add-implicit-args and add-args separately

(def add-implicit-args
  (fn [fd]
    (println "  add-implicit-args fd:" fd)
    (let [args (first fd)]
      (println "  args:" args)
      (cons (vec (cons '&form (cons '&env args))) (next fd)))))

(def add-args
  (fn [acc ds]
    (println "add-args acc:" acc "ds:" ds)
    (if (nil? ds)
      acc
      (let [d (first ds)]
        (println "  d:" d)
        (if (map? d)
          (recur (conj acc d) (next ds))
          (recur (conj acc (add-implicit-args d)) (next ds)))))))

;; Simulate what happens with ([test & body] `(if ~test (do ~@body)))
(def test-args '([test & body] some-body))

(println "test-args:" test-args)
(println "first test-args:" (first test-args))
(println "vector? first:" (vector? (first test-args)))

(def fdecl (list test-args))
(println "fdecl:" fdecl)

(def result (seq (add-args [] fdecl)))
(println "Result:" result)
