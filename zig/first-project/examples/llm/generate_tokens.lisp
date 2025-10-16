;; ============================================================================
;; Interactive GPT-2 Token Generation
;; Usage: Reads initial tokens from stdin, generates new tokens
;; ============================================================================

(include-header "stdio.h")
(include-header "stdlib.h")
(include-header "math.h")
(include-header "string.h")
(link-library "m")
(compiler-flag "-O3")

(declare-fn printf [fmt (Pointer U8)] -> I32)
(declare-fn scanf [fmt (Pointer U8)] -> I32)
(declare-fn malloc [size I32] -> (Pointer U8))
(declare-fn free [ptr (Pointer U8)] -> Nil)
(declare-fn fopen [filename (Pointer U8) mode (Pointer U8)] -> (Pointer U8))
(declare-fn fclose [file (Pointer U8)] -> I32)
(declare-fn fread [ptr (Pointer U8) size I32 count I32 file (Pointer U8)] -> I32)
(declare-fn exit [status I32] -> Nil)
(declare-fn sqrtf [x F32] -> F32)
(declare-fn tanhf [x F32] -> F32)
(declare-fn expf [x F32] -> F32)
(declare-fn fflush [stream (Pointer U8)] -> I32)

;; Just use the same structs and functions from llm.lisp
;; For brevity, I'll create a minimal version that prints tokens one at a time

(def main-fn (: (-> [] I32))
  (fn []
    ;; Read number of initial tokens
    (printf (c-str "Enter number of initial tokens: "))
    (let [num_init (: I32) 0]
      (scanf (c-str "%d") (address-of num_init))

      ;; Read initial tokens
      (let [i (: I32) 0
            tokens (: (Pointer I32)) (allocate-array I32 200)]
        (printf (c-str "Enter %d tokens: ") num_init)
        (while (< i num_init)
          (scanf (c-str "%d") (+ tokens i))
          (set! i (+ i 1)))

        ;; Echo back the tokens
        (printf (c-str "\nTokens: "))
        (set! i 0)
        (while (< i num_init)
          (printf (c-str "%d ") (pointer-index-read tokens i))
          (fflush pointer-null)
          (set! i (+ i 1)))

        (printf (c-str "\n"))
        (deallocate-array tokens)
        0))))

(main-fn)
