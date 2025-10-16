(include-header "stdio.h")
(include-header "stdlib.h")
(include-header "math.h")
(link-library "m")
(compiler-flag "-O3")

(declare-fn printf [fmt (Pointer U8)] -> I32)
(declare-fn malloc [size I32] -> (Pointer U8))

(def main-fn (: (-> [] I32))
  (fn []
    (printf (c-str "Testing 1 token generation timing...\n"))
    0))

(main-fn)
