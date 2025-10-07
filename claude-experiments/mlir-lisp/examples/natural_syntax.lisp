;; Goal: Write code that looks natural and macro-expands to MLIR

;; Basic operators
(defmacro const [value]
  (op arith.constant :attrs {:value value} :results [i32]))

(defmacro + [a b]
  (op arith.addi :operands [a b] :results [i32]))

(defmacro - [a b]
  (op arith.subi :operands [a b] :results [i32]))

(defmacro * [a b]
  (op arith.muli :operands [a b] :results [i32]))

(defmacro <= [a b]
  (op arith.cmpi :attrs {:predicate "sle"} :operands [a b] :results [i1]))

;; Let macro for local bindings
;; (let [x 5] (+ x 1))
;; We'll expand this manually for now as a simple sequential binding
(defmacro let1 [name value body]
  (block let_block []
    value
    body))

;; Test: compute factorial-like expression: 5 * 4 * 3
(defn main [] i32
  (const 5 :as %a)
  (const 4 :as %b)
  (const 3 :as %c)
  (* %a %b :as %temp)
  (* %temp %c :as %result)
  (op func.return :operands [%result]))
