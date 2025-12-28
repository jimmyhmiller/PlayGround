;; Simple file I/O test using pure libc (no Rust FFI)

(require-dialect func)
(require-dialect arith)
(require-dialect llvm)

(link-library :c)

;; External function declarations for libc
(extern-fn malloc (-> [i64] [!llvm.ptr]))
(extern-fn free (-> [!llvm.ptr] []))
(extern-fn fopen (-> [!llvm.ptr !llvm.ptr] [!llvm.ptr]))
(extern-fn fread (-> [!llvm.ptr i64 i64 !llvm.ptr] [i64]))
(extern-fn fclose (-> [!llvm.ptr] [i32]))
(extern-fn printf (-> [!llvm.ptr] [i32]))
(extern-fn printf_1 (-> [!llvm.ptr i64] [i32]))

;; Global string constants
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

    (llvm.mlir.global {:sym_name "fmt_ptr"
                       :linkage 0
                       :global_type !llvm.array<18 x i8>
                       :constant true}
      (region
        (block []
          (def _str (llvm.mlir.constant {:value "File pointer: %p\n\0" :result !llvm.array<18 x i8>}))
          (llvm.return _str))))

    (llvm.mlir.global {:sym_name "fmt_read"
                       :linkage 0
                       :global_type !llvm.array<17 x i8>
                       :constant true}
      (region
        (block []
          (def _str (llvm.mlir.constant {:value "Bytes read: %ld\n\0" :result !llvm.array<17 x i8>}))
          (llvm.return _str))))

    (llvm.mlir.global {:sym_name "fmt_byte"
                       :linkage 0
                       :global_type !llvm.array<13 x i8>
                       :constant true}
      (region
        (block []
          (def _str (llvm.mlir.constant {:value "Byte 0: %ld\n\0" :result !llvm.array<13 x i8>}))
          (llvm.return _str))))))

(defn main [] -> i64
  ;; Get string addresses
  (def path (llvm.mlir.addressof {:global_name @test_path :result !llvm.ptr}))
  (def mode (llvm.mlir.addressof {:global_name @read_mode :result !llvm.ptr}))

  ;; Open file
  (def file (call !llvm.ptr fopen path mode))

  ;; Print file pointer for debugging
  (def fmt_ptr (llvm.mlir.addressof {:global_name @fmt_ptr :result !llvm.ptr}))
  (def file_int (llvm.ptrtoint {:result i64} file))
  (call i32 printf_1 fmt_ptr file_int)

  ;; Allocate buffer (64 bytes)
  (def buf_size (: 64 i64))
  (def buf (call !llvm.ptr malloc buf_size))

  ;; Read from file
  (def bytes_read (call i64 fread buf (: 1 i64) buf_size file))

  ;; Print bytes read
  (def fmt_read (llvm.mlir.addressof {:global_name @fmt_read :result !llvm.ptr}))
  (call i32 printf_1 fmt_read bytes_read)

  ;; Read first byte and print it
  (def first_byte (llvm.load {:result i8} buf))
  (def first_byte_i64 (arith.extui {:result i64} first_byte))
  (def fmt_byte (llvm.mlir.addressof {:global_name @fmt_byte :result !llvm.ptr}))
  (call i32 printf_1 fmt_byte first_byte_i64)

  ;; Close file
  (call i32 fclose file)

  ;; Free buffer
  (call! free buf)

  (func.return (: 0 i64)))