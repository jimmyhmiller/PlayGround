;; Test parts of the real defmacro

(println "Testing loop with recur on (next args):")

;; Test the prefix loop
(def test-args '("docstring" {:foo 1} [x y] (+ x y)))

(println "test-args:" test-args)

(def result
  (loop [p (list 'myname) args test-args]
    (println "p:" p "args:" args)
    (let [f (first args)]
      (println "f:" f)
      (if (string? f)
        (do
          (println "Found string, recurring...")
          (recur (cons f p) (next args)))
        (if (map? f)
          (do
            (println "Found map, recurring...")
            (recur (cons f p) (next args)))
          (do
            (println "Done, returning p")
            p))))))

(println "Result:" result)
