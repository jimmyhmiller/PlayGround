;; Test fdecl loop

(println "Testing fdecl loop:")

(def test-args '("docstring" {:foo 1} [x y] (+ x y)))

(println "test-args:" test-args)

(def fdecl
  (loop [fd test-args]
    (println "fd:" fd)
    (if (string? (first fd))
      (do
        (println "Skipping string...")
        (recur (next fd)))
      (if (map? (first fd))
        (do
          (println "Skipping map...")
          (recur (next fd)))
        (do
          (println "Done!")
          fd)))))

(println "fdecl:" fdecl)

;; Test wrapping in list if vector
(println "Testing vector? check:")
(println "first fdecl:" (first fdecl))
(println "vector? first:" (vector? (first fdecl)))

(def fdecl2 (if (vector? (first fdecl))
              (list fdecl)
              fdecl))

(println "fdecl2:" fdecl2)
