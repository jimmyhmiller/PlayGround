;; Example 8: Macros
;; Demonstrates: Macro definitions, syntax-quote, unquote, auto-gensym

(include-header "stdio.h")
(declare-fn printf [fmt (Pointer U8)] -> I32)

;; Simple macro - add 1
(defmacro add1 [x]
  `(+ ~x 1))

;; Macro with multiple operations
(defmacro square [x]
  `(* ~x ~x))

;; Macro that expands to another macro call
(defmacro double-add1 [x]
  `(add1 (add1 ~x)))

;; Macro using manual gensym for temporary variables
(defmacro let-temp [val body]
  (let [tmp-name (gensym "temp")]
    `(let [~tmp-name (: I32) ~val]
       ~body)))

(def main-fn (: (-> [] I32))
  (fn []
    ;; Test add1 macro
    (let [x (: I32) (add1 5)]
      (printf (c-str "add1(5) = %d\n") x))

    ;; Test square macro
    (let [y (: I32) (square 7)]
      (printf (c-str "square(7) = %d\n") y))

    ;; Test macro composition
    (let [z (: I32) (double-add1 10)]
      (printf (c-str "double-add1(10) = %d\n") z))

    ;; Test computed values
    (let [result (: I32) (+ (add1 3) (square 4))]
      (printf (c-str "(add1 3) + (square 4) = %d\n") result))

    0))

(main-fn)
