;; Minimal test to debug fread behavior

(require-dialect func)
(require-dialect arith)
(require-dialect llvm)
(require-dialect scf)

(link-library :c)

(extern-fn malloc (-> [i64] [!llvm.ptr]))
(extern-fn fopen (-> [!llvm.ptr !llvm.ptr] [!llvm.ptr]))
(extern-fn fread (-> [!llvm.ptr i64 i64 !llvm.ptr] [i64]))
(extern-fn fseek (-> [!llvm.ptr i64 i32] [i32]))
(extern-fn ftell (-> [!llvm.ptr] [i64]))
(extern-fn fclose (-> [!llvm.ptr] [i32]))

;; printf is variadic
(extern-fn printf (-> [!llvm.ptr ...] [i32]))

(module
  (do
    ;; Checkpoint path
    (llvm.mlir.global {:sym_name "checkpoint_path"
                       :linkage 0
                       :global_type !llvm.array<39 x i8>
                       :constant true}
      (region
        (block []
          (def _str_val (llvm.mlir.constant {:value "/home/jimmyhmiller/llm.c/gpt2_124M.bin\0" :result !llvm.array<39 x i8>}))
          (llvm.return _str_val))))

    (llvm.mlir.global {:sym_name "read_mode_str"
                       :linkage 0
                       :global_type !llvm.array<3 x i8>
                       :constant true}
      (region
        (block []
          (def _str (llvm.mlir.constant {:value "rb\0" :result !llvm.array<3 x i8>}))
          (llvm.return _str))))))

;; Simple function that allocates, reads file, returns pointer
(defn test_fread [] -> !llvm.ptr
  ;; Open file
  (def path (llvm.mlir.addressof {:global_name @checkpoint_path :result !llvm.ptr}))
  (def mode (llvm.mlir.addressof {:global_name @read_mode_str :result !llvm.ptr}))
  (def file (call !llvm.ptr fopen path mode))

  ;; Print file pointer
  (def file_int (llvm.ptrtoint {:result i64} file))
  (print "file ptr = 0x%lx\n" file_int)

  ;; Read header (1024 bytes, skip it)
  (def header_size (: 1024 i64))
  (def header_ptr (call !llvm.ptr malloc header_size))
  (def header_count (: 256 i64))
  (def four (: 4 i64))
  (def header_read (call i64 fread header_ptr four header_count file))
  (print "header fread returned: %ld\n" header_read)

  ;; Allocate small buffer: just 100 floats (400 bytes)
  (def num_floats (: 100 i64))
  (def sizeof_f32 (: 4 i64))
  (def buf_size (arith.muli num_floats sizeof_f32))
  (def ptr (call !llvm.ptr malloc buf_size))

  ;; Print the pointer
  (def ptr_int (llvm.ptrtoint {:result i64} ptr))
  (print "In test_fread: ptr = 0x%lx\n" ptr_int)

  ;; Write known values first
  (def test_val (: -999.0 f32))
  (llvm.store test_val ptr)
  (def check1 (llvm.load {:result f32} ptr))
  (def check1_64 (arith.extf {:result f64} check1))
  (def z (: 0.0 f64))
  (print "Before fread: val[0]=%f\n" check1_64 z z z)

  ;; Now call fread
  (def read_count (call i64 fread ptr sizeof_f32 num_floats file))
  (print "fread returned: %ld\n" read_count)

  ;; Read back to verify
  (def read0 (llvm.load {:result f32} ptr))
  (def read0_64 (arith.extf {:result f64} read0))
  (def idx1 (: 1 i64))
  (def ptr1 (ptr-at f32 ptr idx1))
  (def read1 (llvm.load {:result f32} ptr1))
  (def read1_64 (arith.extf {:result f64} read1))
  (print "After fread in function: val[0]=%f, val[1]=%f\n" read0_64 read1_64 z z)

  ;; Don't close file - see if that helps
  ;; (def close_result (call i32 fclose file))

  ;; Return the pointer
  (func.return ptr))

(defn main [] -> i64
  ;; Call the function and get the pointer
  (def ptr (call !llvm.ptr test_fread))

  ;; Print the pointer
  (def ptr_int (llvm.ptrtoint {:result i64} ptr))
  (print "In main: ptr = 0x%lx\n" ptr_int)

  ;; Read the values
  (def read0 (llvm.load {:result f32} ptr))
  (def read0_64 (arith.extf {:result f64} read0))
  (def idx1 (: 1 i64))
  (def ptr1 (ptr-at f32 ptr idx1))
  (def read1 (llvm.load {:result f32} ptr1))
  (def read1_64 (arith.extf {:result f64} read1))
  (def z (: 0.0 f64))
  (print "In main: val[0]=%f, val[1]=%f\n" read0_64 read1_64 z z)

  (func.return (: 0 i64)))
