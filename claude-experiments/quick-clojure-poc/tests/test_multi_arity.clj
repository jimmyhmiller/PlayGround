;; Test multi-arity functions

;; Simple multi-arity function
(def greet
  (fn
    ([] 0)
    ([x] x)
    ([x y] (+ x y))))

;; Test arity 0
(greet)

;; Test arity 1
(greet 42)

;; Test arity 2
(greet 10 20)
