; Minimal defstruct test

(require-dialect func)
(require-dialect llvm)
(require-dialect arith)

(link-library :c)

; Declare malloc
(func.func {:sym_name "malloc" :function_type (-> [i64] [!llvm.ptr]) :sym_visibility "private"})

; Define a Point struct
(defstruct Point [x i64] [y i64])

; Main function using defn
(defn main [] -> i64
  (def p (new Point))
  (Point/x! p (: 10 i64))
  (Point/y! p (: 32 i64))
  (def sum (arith.addi (Point/x p) (Point/y p)))
  (func.return sum))
