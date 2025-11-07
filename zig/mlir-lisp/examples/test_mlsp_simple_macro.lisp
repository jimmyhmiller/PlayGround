;; Test: Can we use mlsp.identifier with a string literal?
;;
;; This tests the simplest case: creating an identifier with a compile-time known string.
;; The mlsp.identifier operation expects a string literal attribute.

(defn simpleIdMacro [(: %args_ptr !llvm.ptr)] !llvm.ptr

  ;; Try to create an identifier with a string literal
  (op %id (: !llvm.ptr) (mlsp.identifier {:value "test"}))

  (return %id))
