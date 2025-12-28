;; Simple file I/O test using llvm.call for variadic printf

(require-dialect func)
(require-dialect arith)
(require-dialect llvm)

(link-library :c)

;; Use llvm.func for external declarations
(module
  (do
    ;; Declare libc functions using llvm.func
    (llvm.func {:sym_name "malloc"
                :function_type (-> [i64] [!llvm.ptr])
                :linkage 10})
    (llvm.func {:sym_name "free"
                :function_type (-> [!llvm.ptr] [])
                :linkage 10})
    (llvm.func {:sym_name "fopen"
                :function_type (-> [!llvm.ptr !llvm.ptr] [!llvm.ptr])
                :linkage 10})
    (llvm.func {:sym_name "fread"
                :function_type (-> [!llvm.ptr i64 i64 !llvm.ptr] [i64])
                :linkage 10})
    (llvm.func {:sym_name "fclose"
                :function_type (-> [!llvm.ptr] [i32])
                :linkage 10})
    ;; printf is variadic - use llvm.func with vararg_
    (llvm.func {:sym_name "printf"
                :function_type (-> [!llvm.ptr] [i32])
                :vararg_ true
                :linkage 10})

    ;; String constants
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

  ;; Open file using llvm.call
  (def file (llvm.call {:callee @fopen :result !llvm.ptr} path mode))

  ;; Print file pointer
  (def fmt_ptr (llvm.mlir.addressof {:global_name @fmt_ptr :result !llvm.ptr}))
  (def file_int (llvm.ptrtoint {:result i64} file))
  (llvm.call {:callee @printf :result i32} fmt_ptr file_int)

  ;; Allocate buffer
  (def buf_size (: 64 i64))
  (def buf (llvm.call {:callee @malloc :result !llvm.ptr} buf_size))

  ;; Read from file
  (def one (: 1 i64))
  (def bytes_read (llvm.call {:callee @fread :result i64} buf one buf_size file))

  ;; Print bytes read
  (def fmt_read (llvm.mlir.addressof {:global_name @fmt_read :result !llvm.ptr}))
  (llvm.call {:callee @printf :result i32} fmt_read bytes_read)

  ;; Read first byte and print
  (def first_byte (llvm.load {:result i8} buf))
  (def first_byte_i64 (arith.extui {:result i64} first_byte))
  (def fmt_byte (llvm.mlir.addressof {:global_name @fmt_byte :result !llvm.ptr}))
  (llvm.call {:callee @printf :result i32} fmt_byte first_byte_i64)

  ;; Close and free
  (llvm.call {:callee @fclose :result i32} file)
  (llvm.call {:callee @free} buf)

  (func.return (: 0 i64)))