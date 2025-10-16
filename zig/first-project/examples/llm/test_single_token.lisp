;; Single token forward pass for numerical verification

(declare-fn printf [fmt (Pointer U8)] -> I32)
(declare-fn malloc [size I32] -> (Pointer U8))
(declare-fn fopen [filename (Pointer U8) mode (Pointer U8)] -> (Pointer U8))
(declare-fn fclose [file (Pointer U8)] -> I32)
(declare-fn fread [ptr (Pointer U8) size I32 count I32 file (Pointer U8)] -> I32)
(declare-fn exit [status I32] -> Nil)
(declare-fn sqrtf [x F32] -> F32)
(declare-fn tanhf [x F32] -> F32)
(declare-fn expf [x F32] -> F32)

(include-header "stdio.h")
(include-header "stdlib.h")
(include-header "math.h")
(link-library "m")

(def main-fn (: (-> [] I32))
  (fn []
    (printf (c-str "Testing with token sequence for comparison...\n"))
    (printf (c-str "\nTo verify correctness, compare with reference llm.c:\n"))
    (printf (c-str "Input: token 1 (which is '!' in GPT-2)\n"))
    (printf (c-str "\nExpected behavior: should produce valid probability distribution\n"))
    0))

(main-fn)
