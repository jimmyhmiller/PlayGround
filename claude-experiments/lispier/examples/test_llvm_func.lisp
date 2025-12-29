;; Test llvm.func declaration only

(require-dialect func)
(require-dialect llvm)

;; printf is variadic
(extern-fn printf (-> [!llvm.ptr ...] [i32]))

(defn main [] -> i64
  (func.return (: 0 i64)))
