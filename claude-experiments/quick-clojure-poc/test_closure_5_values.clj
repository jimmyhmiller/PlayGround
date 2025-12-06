;; Test closure with 5 captured values (previous limit)
(def make-closure-5
  (fn [a b c d e]
    (fn []
      (+ a (+ b (+ c (+ d e)))))))

(def closure-5 (make-closure-5 1 2 3 4 5))
(closure-5)
