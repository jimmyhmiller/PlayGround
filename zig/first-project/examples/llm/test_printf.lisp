;; Test printf with format specifiers

(include-header "stdio.h")

(extern-fn printf [fmt (Pointer U8)] -> I32)

(def main-fn (: (-> [] I32))
  (fn []
    (printf (c-str "Testing printf with format specifiers:\n"))

    ;; Test integer
    (let [x (: I32) 42]
      (printf (c-str "Integer: %d\n") x)
      nil)

    ;; Test float
    (let [y (: F32) 3.14159]
      (printf (c-str "Float: %f\n") y)
      nil)

    ;; Test multiple values
    (let [a (: I32) 10
          b (: F32) 2.5]
      (printf (c-str "Int=%d, Float=%f\n") a b)
      nil)

    ;; Test array values
    (let [arr (: (Pointer F32)) (allocate-array F32 3)]
      (pointer-index-write! arr 0 1.1)
      (pointer-index-write! arr 1 2.2)
      (pointer-index-write! arr 2 3.3)

      (printf (c-str "Array: [%f, %f, %f]\n")
              (pointer-index-read arr 0)
              (pointer-index-read arr 1)
              (pointer-index-read arr 2))

      (deallocate-array arr)
      nil)

    0))

(main-fn)
