;; ============================================================================
;; GPT-2 Text Generation - Reads prompt tokens from stdin
;; ============================================================================

(include-header "stdio.h")
(include-header "stdlib.h")
(include-header "math.h")
(link-library "m")
(compiler-flag "-O3")

(declare-fn printf [fmt (Pointer U8)] -> I32)
(declare-fn scanf_int [fmt (Pointer U8) ptr (Pointer I32)] -> I32)
(declare-fn getchar [] -> I32)
(declare-fn atoi [str (Pointer U8)] -> I32)
(declare-fn malloc [size I32] -> (Pointer U8))
(declare-fn free [ptr (Pointer U8)] -> Nil)
(declare-fn fopen [filename (Pointer U8) mode (Pointer U8)] -> (Pointer U8))
(declare-fn fclose [file (Pointer U8)] -> I32)
(declare-fn fread [ptr (Pointer U8) size I32 count I32 file (Pointer U8)] -> I32)
(declare-fn exit [status I32] -> Nil)
(declare-fn sqrtf [x F32] -> F32)
(declare-fn tanhf [x F32] -> F32)
(declare-fn expf [x F32] -> F32)

;; Include all the GPT-2 implementation from llm.lisp
;; For now, let me just create a minimal version that shows the concept

(def GPT2Config (: Type)
  (Struct
    [max_seq_len I32]
    [vocab_size I32]
    [padded_vocab_size I32]
    [num_layers I32]
    [num_heads I32]
    [channels I32]))

(def main-fn (: (-> [] I32))
  (fn []
    (printf (c-str "Reading prompt tokens from stdin...\n"))
    (printf (c-str "Enter number of prompt tokens: "))

    (let [num_prompt_tokens (: I32) 0]
      (scanf (c-str "%d") (address-of num_prompt_tokens))

      (printf (c-str "Enter %d token IDs (space-separated): ") num_prompt_tokens)

      (let [tokens (: (Pointer I32)) (allocate-array I32 200)
            i (: I32) 0]

        ;; Read tokens from stdin
        (while (< i num_prompt_tokens)
          (scanf (c-str "%d") (+ tokens i))
          (set! i (+ i 1)))

        ;; Echo back what we read
        (printf (c-str "\nRead %d tokens: [") num_prompt_tokens)
        (set! i 0)
        (while (< i num_prompt_tokens)
          (if (< i (- num_prompt_tokens 1))
            (printf (c-str "%d, ") (pointer-index-read tokens i))
            (printf (c-str "%d") (pointer-index-read tokens i)))
          (set! i (+ i 1)))
        (printf (c-str "]\n"))

        ;; TODO: Actually run generation with these tokens
        (printf (c-str "\nGeneration would happen here with your custom prompt!\n"))

        (deallocate-array tokens)
        0))))

(main-fn)
