;; Test single forward pass timing
(include-header "stdio.h")
(include-header "stdlib.h")
(include-header "math.h")
(include-header "time.h")
(link-library "m")
(compiler-flag "-O3")

(declare-fn printf [fmt (Pointer U8)] -> I32)
(declare-fn clock [] -> I64)

(def main-fn (: (-> [] I32))
  (fn []
    (let [start (: I64) (clock)]
      (printf (c-str "Simulating forward pass...\n"))
      (let [end (: I64) (clock)]
        (printf (c-str "Time: %ld clock ticks\n") (- end start))))
    0))

(main-fn)
