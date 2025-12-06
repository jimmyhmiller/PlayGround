;; Test closure with 7 captured values (previously would have failed at 5!)
(def make-closure-7
  (fn [a b c d e f g]
    (fn []
      (+ a (+ b (+ c (+ d (+ e (+ f g)))))))))

(def closure-7 (make-closure-7 1 2 3 4 5 6 7))
(closure-7)
