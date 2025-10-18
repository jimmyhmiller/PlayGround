;; Test if 'and' supports multiple arguments

(include-header "stdio.h")
(declare-fn printf [fmt (Pointer U8)] -> I32)

(def test-and-two-args (: (-> [] I32))
  (fn []
    (if (and true true)
        (printf (c-str "Two args: true\n"))
        (printf (c-str "Two args: false\n")))
    0))

(def test-and-six-args (: (-> [] I32))
  (fn []
    (if (and true true true true true true)
        (printf (c-str "Six args: true\n"))
        (printf (c-str "Six args: false\n")))
    0))

(def main-fn (: (-> [] I32))
  (fn []
    (test-and-two-args)
    (test-and-six-args)
    0))

(main-fn)
