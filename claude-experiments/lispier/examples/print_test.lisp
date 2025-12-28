; Test the new print feature
(require-dialect arith)
(require-dialect scf)
(require-dialect func)
(require-dialect llvm)

(link-library :c)

; External printf declaration (no format args version)
(extern-fn printf (-> [!llvm.ptr] [i32]))

(defn main [] -> i64
  ; Simple prints without format args work!
  (print "Hello, World!\n")
  (println "This is a test")
  (println "")
  (println "Done!")
  (func.return (: 0 i64)))

; NOTE: print-i64 and format args require variadic printf support
; which is not yet implemented. For printing numbers, use the
; character-by-character approach with putchar, as shown in
; examples/binary_trees_benchmark.lisp
