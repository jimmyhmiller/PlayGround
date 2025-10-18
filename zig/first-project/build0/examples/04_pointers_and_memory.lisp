;; Example 4: Pointers and Memory Management
;; Demonstrates: Heap allocation, pointers, dereferencing, pointer writes, address-of

(include-header "stdio.h")
(declare-fn printf [fmt (Pointer U8)] -> I32)

;; Define Point struct at top level
(def Point (: Type) (Struct [x I32] [y I32]))

(def main-fn (: (-> [] I32))
  (fn []
    ;; Demo 1: Allocate integer and modify through pointer
    (let [ptr (: (Pointer I32)) (allocate I32 42)]
      (let [_ (: I32) (printf (c-str "Initial value: %d\n") (dereference ptr))]
        (let [_ (: Nil) (pointer-write! ptr 100)]
          (let [_ (: I32) (printf (c-str "After write: %d\n") (dereference ptr))]
            ;; Demo 2: Allocate struct and modify fields
            (let [point-ptr (: (Pointer Point)) (allocate Point (Point 10 20))]
              (let [_ (: I32) (printf (c-str "Point: (%d, %d)\n")
                                     (pointer-field-read point-ptr x)
                                     (pointer-field-read point-ptr y))]
                (let [_ (: Nil) (pointer-field-write! point-ptr x 30)]
                  (let [_ (: Nil) (pointer-field-write! point-ptr y 40)]
                    (printf (c-str "Modified Point: (%d, %d)\n")
                            (pointer-field-read point-ptr x)
                            (pointer-field-read point-ptr y))
                    0))))))))))

(main-fn)
