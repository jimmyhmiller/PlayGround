;; Test vec on a Cons with PersistentVector rest

(println "Step 1: Create a vector")
(def v [1 2 3])
(println "v:" v)

(println "Step 2: Create a cons with vector as rest")
(def c (cons 'a v))
(println "c:" c)

(println "Step 3: Try to call vec on the cons")
(println "calling vec...")
(def result (vec c))
(println "result:" result)

(println "Done!")
