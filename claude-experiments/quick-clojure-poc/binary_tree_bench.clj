(do
  (deftype Node [left right])

  (def make-tree
    (fn [depth]
      (if (= depth 0)
        (Node. nil nil)
        (Node. (make-tree (- depth 1)) (make-tree (- depth 1))))))

  (def check-tree
    (fn [node]
      (if (= (.-left node) nil)
        1
        (+ 1 (+ (check-tree (.-left node)) (check-tree (.-right node)))))))

  (def run-iteration
    (fn [depth]
      (check-tree (make-tree depth))))

  (run-iteration 15))