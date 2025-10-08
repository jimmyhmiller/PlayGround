;; Demo of custom operations in our own namespace
;; We'll emit operations like "lisp.if", "lisp.add", etc.
;; Then show how to lower them to standard MLIR

(defn demo [] i32
  42)

(defn main [] i32
  (demo))
