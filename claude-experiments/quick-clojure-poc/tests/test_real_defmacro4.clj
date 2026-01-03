(def defmacro2
  (fn [&form &env name & args]
    (println "defmacro2 called with name:" name)
    (println "args:" args)
    (let [prefix (loop [p (list name) args args]
                   (let [f (first args)]
                     (if (string? f)
                       (recur (cons f p) (next args))
                       (if (map? f)
                         (recur (cons f p) (next args))
                         p))))
          _ (println "prefix:" prefix)
          fdecl (loop [fd args]
                  (if (string? (first fd))
                    (recur (next fd))
                    (if (map? (first fd))
                      (recur (next fd))
                      fd)))
          _ (println "fdecl before:" fdecl)
          fdecl (if (vector? (first fdecl))
                  (list fdecl)
                  fdecl)
          _ (println "fdecl after list:" fdecl)
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
          _ (println "fdecl final:" fdecl)
          decl (loop [p prefix d fdecl]
                 (if p
                   (recur (next p) (cons (first p) d))
                   d))
          _ (println "decl:" decl)]
      (list 'do
            (cons 'defn decl)
            (list '__set_macro! (list 'var name))
            (list 'var name)))))

(__set_macro! (var defmacro2))

(println "defmacro2 defined!")

(defmacro2 my-when [test & body]
  `(if ~test (do ~@body)))
