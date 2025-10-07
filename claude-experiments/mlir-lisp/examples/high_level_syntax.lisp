;; High-level syntax macros for natural code

;; Arithmetic operators
(defmacro + [a b]
  (op arith.addi :operands [a b] :results [i32]))

(defmacro - [a b]
  (op arith.subi :operands [a b] :results [i32]))

(defmacro * [a b]
  (op arith.muli :operands [a b] :results [i32]))

;; Comparison operators
(defmacro <= [a b]
  (op arith.cmpi :attrs {:predicate "sle"} :operands [a b] :results [i1]))

(defmacro < [a b]
  (op arith.cmpi :attrs {:predicate "slt"} :operands [a b] :results [i1]))

(defmacro > [a b]
  (op arith.cmpi :attrs {:predicate "sgt"} :operands [a b] :results [i1]))

;; Constant macro
(defmacro const [value]
  (op arith.constant :attrs {:value value} :results [i32]))

;; Test: compute (5 + 3) * 2 - 1
(defn main [] i32
  (const 5 :as %five)
  (const 3 :as %three)
  (const 2 :as %two)
  (const 1 :as %one)

  ;; Use high-level operators - need to manually add :as for now
  (+ %five %three :as %sum)
  (* %sum %two :as %product)
  (- %product %one :as %result)

  (op func.return :operands [%result]))
