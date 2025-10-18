;; Test to demonstrate the constraint:
;; Multiple expressions work in function bodies but NOT in let bodies

(include-header "stdio.h")
(declare-fn printf [fmt (Pointer U8)] -> I32)

;; This WORKS - multiple statements in function body
(def test-function-body (: (-> [] I32))
  (fn []
    (let [x (: I32) 0]
      (while (< x 3)
        (set! x (+ x 1)))
      (printf (c-str "After while: x = %d\n") x)
      42)))  ; Return value

;; This FAILS - multiple expressions in let body
(def test-let-body (: (-> [] I32))
  (fn []
    (let [result (: I32)
          (let [x (: I32) 0]
            (while (< x 3)        ; Expression 1: returns Nil
              (set! x (+ x 1)))
            x)]                   ; Expression 2: ERROR!
      (printf (c-str "Result: %d\n") result)
      0)))

(def main-fn (: (-> [] I32))
  (fn []
    (printf (c-str "Test 1: Function body with while (should work)\n"))
    (test-function-body)

    (printf (c-str "\nTest 2: Let body with while (should fail)\n"))
    (test-let-body)

    0))

(main-fn)
