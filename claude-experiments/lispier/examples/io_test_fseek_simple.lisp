;; Simple fseek/ftell test

(require-dialect func)
(require-dialect arith)
(require-dialect llvm)

(link-library :c)

;; External function declarations
(extern-fn fopen (-> [!llvm.ptr !llvm.ptr] [!llvm.ptr]))
(extern-fn fclose (-> [!llvm.ptr] [i32]))
(extern-fn fseek (-> [!llvm.ptr i64 i32] [i32]))
(extern-fn ftell (-> [!llvm.ptr] [i64]))
(extern-fn printf (-> [!llvm.ptr ...] [i32]))

;; String constants
(module
  (do
    (llvm.mlir.global {:sym_name "test_path"
                       :linkage 0
                       :global_type !llvm.array<19 x i8>
                       :constant true}
      (region
        (block []
          (def _str (llvm.mlir.constant {:value "/tmp/test_file.txt\0" :result !llvm.array<19 x i8>}))
          (llvm.return _str))))

    (llvm.mlir.global {:sym_name "read_mode"
                       :linkage 0
                       :global_type !llvm.array<2 x i8>
                       :constant true}
      (region
        (block []
          (def _str (llvm.mlir.constant {:value "r\0" :result !llvm.array<2 x i8>}))
          (llvm.return _str))))

    (llvm.mlir.global {:sym_name "fmt_size"
                       :linkage 0
                       :global_type !llvm.array<16 x i8>
                       :constant true}
      (region
        (block []
          (def _str (llvm.mlir.constant {:value "File size: %ld\n\0" :result !llvm.array<16 x i8>}))
          (llvm.return _str))))))

(defn main [] -> i64
  ;; Get string addresses
  (def path (llvm.mlir.addressof {:global_name @test_path :result !llvm.ptr}))
  (def mode (llvm.mlir.addressof {:global_name @read_mode :result !llvm.ptr}))

  ;; Open file
  (def file (call !llvm.ptr fopen path mode))

  ;; Get file size using fseek/ftell
  ;; fseek to end (SEEK_END = 2)
  (call i32 fseek file (: 0 i64) (: 2 i32))

  ;; ftell returns current position = file size
  (def file_size (call i64 ftell file))

  ;; Print file size
  (def fmt_size (llvm.mlir.addressof {:global_name @fmt_size :result !llvm.ptr}))
  (vararg-call i32 printf (-> [!llvm.ptr ...] [i32]) fmt_size file_size)

  ;; Close file
  (call i32 fclose file)

  (func.return (: 0 i64)))
