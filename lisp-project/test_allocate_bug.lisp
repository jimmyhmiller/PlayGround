;; Minimal reproducing case for allocate bug
;; The allocate expression should copy the value into allocated memory
;; Currently it just allocates but doesn't copy

(ns test-allocate)

(include-header "stdio.h")
(include-header "stdlib.h")

(declare-fn malloc [size I32] -> (Pointer U8))
(declare-fn printf [fmt (Pointer U8)] -> I32)

;; Simple struct to test with
(def Point (: Type)
  (Struct
    [x I32]
    [y I32]))

(def test-allocate (: (-> [] I32))
  (fn []
    ;; Create a Point value on the stack
    (let [p (: Point) (Point 42 100)]

      ;; BUG: This should allocate memory AND copy p into it
      ;; Currently it only allocates, leaving uninitialized memory
      (let [p-ptr (: (Pointer Point)) (allocate Point p)]

        ;; Read back the x field - should be 42
        (let [x-val (: I32) (pointer-field-read p-ptr x)]
          (printf (c-str "x value: %d (expected 42)\n") x-val)

          ;; If allocate worked correctly, this should print 42
          ;; If it's broken, this will be garbage/0
          x-val)))))

(test-allocate)
