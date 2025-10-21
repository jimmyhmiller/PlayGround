;; Minimal reproducing case for function pointer cast syntax error

(ns test-fn-ptr)

(include-header "stdio.h")

(declare-fn printf [fmt (Pointer U8)] -> I32)

(def test-fn-ptr (: (-> [] I32))
  (fn []
    ;; Simulate getting a void* from some API
    (let [void-ptr (: (Pointer Nil)) (cast (Pointer Nil) 0)]

      ;; BUG: This generates invalid C syntax for function pointer
      ;; The compiler generates: int32_t (*)() main_fn = ((int32_t (*)())void_ptr);
      ;; Which is a syntax error - can't declare and initialize like this
      (let [fn-ptr (: (-> [] I32)) (cast (-> [] I32) void-ptr)]

        (printf (c-str "If this compiles, the bug is fixed\n"))
        0))))

(test-fn-ptr)
