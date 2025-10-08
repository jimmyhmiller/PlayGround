;; Example: Using the lisp Dialect
;; =================================
;; Import our custom dialect (like Racket's #lang)

#lang lisp

;; Import the transformation pipeline
(import-transform lower-lisp-to-arith)
(import-transform constant-fold-add)

;; Now we can write code using high-level lisp dialect operations!
;; These will automatically emit lisp.* operations

(defn compute-expression [] i32
  ;; This emits lisp.constant, lisp.add, lisp.mul, lisp.sub
  (+ (* 10 20) (- 30 5)))

(defn with-constant-folding [] i32
  ;; This will be folded: (+ 10 20) -> 30 at compile time
  (+ 10 20))

(defn nested-expression [] i32
  ;; Complex expression
  (* (+ 5 3) (- 10 2)))

;; Apply transformations
(apply-transform lower-lisp-to-arith)
(apply-transform constant-fold-add)

(defn main [] i32
  (compute-expression))
