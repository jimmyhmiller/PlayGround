;; Test llvm.func with non-vararg type

(require-dialect func)
(require-dialect llvm)

;; Non-vararg function - uses func.func
(extern-fn malloc (-> [i64] [!llvm.ptr]))

(defn main [] -> i64
  (func.return (: 0 i64)))
