;; Example 1: Hello World
;; Demonstrates: Basic program structure, C FFI, string literals

(include-header "stdio.h")
(declare-fn printf [fmt (Pointer U8)] -> I32)

(def main-fn (: (-> [] I32))
  (fn []
    (printf (c-str "Hello, World!\n"))
    0))

(main-fn)
