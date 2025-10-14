;; Simple execution test
;; Define a function and execute it

(defn compute [] i32
  (+ (* 10 20) 30))

(println "\n🚀 Executing compute function...")
(jit-execute compute)

(println "\n✅ Complete!")
