;; Simple printf test

(require-dialect func)
(require-dialect arith)
(require-dialect llvm)

(link-library :c)

;; printf is variadic
(extern-fn printf (-> [!llvm.ptr ...] [i32]))

(module
  (do
    (llvm.mlir.global {:sym_name "hello"
                       :linkage 0
                       :global_type !llvm.array<13 x i8>
                       :constant true}
      (region
        (block []
          (def _str (llvm.mlir.constant {:value "Hello World\n\0" :result !llvm.array<13 x i8>}))
          (llvm.return _str))))))

(defn main [] -> i64
  (def fmt (llvm.mlir.addressof {:global_name @hello :result !llvm.ptr}))
  (vararg-call i32 printf (-> [!llvm.ptr ...] [i32]) fmt)
  (func.return (: 0 i64)))
