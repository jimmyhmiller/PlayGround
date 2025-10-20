;; Minimal reproducing case for allocate bug with OPAQUE types
;; The bug: allocate doesn't copy opaque types properly

(ns test-allocate-opaque)

(include-header "stdio.h")
(include-header "stdlib.h")

(declare-fn malloc [size I32] -> (Pointer U8))
(declare-fn printf [fmt (Pointer U8)] -> I32)

;; Declare an OPAQUE type (like MlirOperationState)
(declare-type OpaqueState)

;; Declare a function that creates an opaque value
(declare-fn create_opaque_state [] -> OpaqueState)

(def test-allocate-opaque (: (-> [] I32))
  (fn []
    ;; Create an opaque value on the stack
    (let [state (: OpaqueState) (create_opaque_state)]

      ;; BUG: This should allocate AND copy state into it
      ;; For opaque types, this is currently broken
      (let [state-ptr (: (Pointer OpaqueState)) (allocate OpaqueState state)]

        (printf (c-str "If this compiles without the copy, the bug is reproduced\n"))
        0))))

(test-allocate-opaque)
