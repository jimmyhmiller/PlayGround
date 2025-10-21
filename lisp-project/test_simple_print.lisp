(ns test-print)

(include-header "stdio.h")
(declare-fn printf [fmt (Pointer U8)] -> I32)
(declare-fn fflush [stream (Pointer Nil)] -> I32)
(declare-var stderr (Pointer Nil))

(def main-test (: (-> [] I32))
  (fn []
    (printf (c-str "Hello from test\n"))
    (fflush stderr)
    0))

(main-test)
