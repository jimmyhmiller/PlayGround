;; Test add-implicit-args and add-args

(println "Testing add-implicit-args:")

(def add-implicit-args
  (fn [fd]
    (let [args (first fd)]
      (println "  fd:" fd)
      (println "  args:" args)
      (cons (vec (cons '&form (cons '&env args))) (next fd)))))

(def test-fd '([x y] (+ x y)))
(println "test-fd:" test-fd)
(println "result:" (add-implicit-args test-fd))

(println)
(println "Testing add-args:")

(def add-args
  (fn [acc ds]
    (println "  add-args acc:" acc "ds:" ds)
    (if (nil? ds)
      (do
        (println "  ds is nil, returning acc")
        acc)
      (let [d (first ds)]
        (println "  d:" d)
        (if (map? d)
          (do
            (println "  d is map, conj and recur")
            (recur (conj acc d) (next ds)))
          (do
            (println "  d is not map, add-implicit-args and recur")
            (recur (conj acc (add-implicit-args d)) (next ds))))))))

(def fdecl '(([x y] (+ x y))))
(println "fdecl:" fdecl)
(println "add-args result:" (add-args [] fdecl))
