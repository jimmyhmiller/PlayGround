;; Simple File I/O Test - Just verify fopen/fclose works

(include-header "stdio.h")

;; File I/O (FILE* is just a pointer, use Void for opaque type)
(extern-fn fopen [filename String mode String] -> (Pointer Void))
(extern-fn fclose [stream (Pointer Void)] -> I32)
(extern-fn fprintf [stream (Pointer Void) fmt String] -> I32)
(extern-fn printf [fmt String] -> I32)

(def test_file_ops (: (-> [] I32))
  (fn []
    (printf "=== Testing File I/O ===\n")

    ;; Test writing to file
    (let [f_out (: (Pointer Void)) (fopen "/tmp/test_llm.txt" "w")]
      (if (pointer-equal? f_out pointer-null)
        (printf "ERROR: Failed to open file for writing\n")
        (do
          (fprintf f_out "Hello from Lisp!\n")
          (fprintf f_out "File I/O works!\n")
          (fclose f_out)
          (printf "Successfully wrote to /tmp/test_llm.txt\n")))
      nil)

    ;; Test opening for reading (demonstrates fopen works both ways)
    (let [f_in (: (Pointer Void)) (fopen "/tmp/test_llm.txt" "r")]
      (if (pointer-equal? f_in pointer-null)
        (printf "ERROR: Failed to open file for reading\n")
        (do
          (fclose f_in)
          (printf "Successfully opened file for reading\n")))
      nil)

    (printf "File I/O test completed!\n")
    (printf "Check /tmp/test_llm.txt to see the output\n")
    0))

(def main-fn (: (-> [] I32))
  (fn []
    (test_file_ops)))

(main-fn)
