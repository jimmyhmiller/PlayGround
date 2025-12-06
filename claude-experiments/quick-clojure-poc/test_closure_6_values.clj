;; Test closure with 6 captured values
(def make-closure-6
  (fn [a b c d e f]
    (fn []
      (+ a (+ b (+ c (+ d (+ e f))))))))

(def closure-6 (make-closure-6 1 2 3 4 5 6))
(closure-6)
