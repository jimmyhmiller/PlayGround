;; Test full decl construction

(def add-implicit-args
  (fn [fd]
    (let [args (first fd)]
      (cons (vec (cons '&form (cons '&env args))) (next fd)))))

(def add-args
  (fn [acc ds]
    (if (nil? ds)
      acc
      (let [d (first ds)]
        (if (map? d)
          (recur (conj acc d) (next ds))
          (recur (conj acc (add-implicit-args d)) (next ds)))))))

(def test-args '([test & body] some-body))
(def name 'my-when)

(def prefix (list name))
(println "prefix:" prefix)

(def fdecl (list test-args))
(println "fdecl list:" fdecl)

(def fdecl-transformed (seq (add-args [] fdecl)))
(println "fdecl-transformed:" fdecl-transformed)
(println "first fdecl-transformed:" (first fdecl-transformed))

;; Now build decl by prepending prefix
(def decl
  (loop [p prefix d fdecl-transformed]
    (if p
      (recur (next p) (cons (first p) d))
      d)))

(println "decl:" decl)
(println "first decl:" (first decl))
(println "second decl:" (second decl))
