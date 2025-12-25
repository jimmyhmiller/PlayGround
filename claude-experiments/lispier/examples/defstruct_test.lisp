; Test defstruct - defines accessor macros for structs

(require-dialect func)
(require-dialect llvm)
(require-dialect arith)

(link-library :c)

; Declare malloc for heap allocation
(func.func {:sym_name "malloc" :function_type (-> [i64] [!llvm.ptr]) :sym_visibility "private"})

; Define a Point struct with x and y fields
(defstruct Point [x i64] [y i64])

; Main function to test struct operations
(defn main [] -> i64
  ; Allocate a new Point
  (def p (new Point))

  ; Set fields
  (Point/x! p (: 10 i64))
  (Point/y! p (: 32 i64))

  ; Read and add them: 10 + 32 = 42
  (def sum (arith.addi (Point/x p) (Point/y p)))

  (func.return sum))
