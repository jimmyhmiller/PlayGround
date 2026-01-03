;; Test the real Clojure defmacro (without metadata)

(def defmacro2
  (fn [&form &env name & args]
    (let [prefix (loop [p (list name) args args]
                   (let [f (first args)]
                     (if (string? f)
                       (recur (cons f p) (next args))
                       (if (map? f)
                         (recur (cons f p) (next args))
                         p))))
          fdecl (loop [fd args]
                  (if (string? (first fd))
                    (recur (next fd))
                    (if (map? (first fd))
                      (recur (next fd))
                      fd)))
          fdecl (if (vector? (first fdecl))
                  (list fdecl)
                  fdecl)
          add-implicit-args (fn [fd]
                    (let [args (first fd)]
                      (cons (vec (cons '&form (cons '&env args))) (next fd))))
          add-args (fn [acc ds]
                     (if (nil? ds)
                       acc
                       (let [d (first ds)]
                         (if (map? d)
                           (conj acc d)
                           (recur (conj acc (add-implicit-args d)) (next ds))))))
          fdecl (seq (add-args [] fdecl))
          decl (loop [p prefix d fdecl]
                 (if p
                   (recur (next p) (cons (first p) d))
                   d))]
      (list 'do
            (cons `defn decl)
            (list '__set_macro! (list 'var name))
            (list 'var name)))))

(__set_macro! (var defmacro2))

(println "Testing real defmacro:")

;; Test simple macro
(defmacro2 my-when [test & body]
  `(if ~test (do ~@body)))

(println "Calling my-when true:")
(my-when true (println "yes!"))

(println "Done!")