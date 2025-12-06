;; Test closure with 10 captured values (previously impossible!)
(def make-closure-10
  (fn [a b c d e f g h i j]
    (fn []
      (+ a (+ b (+ c (+ d (+ e (+ f (+ g (+ h (+ i j))))))))))))

(def closure-10 (make-closure-10 1 2 3 4 5 6 7 8 9 10))
(closure-10)
