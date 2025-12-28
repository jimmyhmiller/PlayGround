;; Minimal test - fread into buffer stored in global

(require-dialect func)
(require-dialect arith)
(require-dialect llvm)
(require-dialect scf)

(link-library :c)

(extern-fn malloc (-> [i64] [!llvm.ptr]))
(extern-fn fopen (-> [!llvm.ptr !llvm.ptr] [!llvm.ptr]))
(extern-fn fread (-> [!llvm.ptr i64 i64 !llvm.ptr] [i64]))
(extern-fn fclose (-> [!llvm.ptr] [i32]))
(extern-fn printf (-> [!llvm.ptr] [i32]))
(extern-fn printf_1 (-> [!llvm.ptr i64] [i32]))
(extern-fn printf_4 (-> [!llvm.ptr f64 f64 f64 f64] [i32]))

(module
  (do
    ;; Global to hold the buffer pointer
    (llvm.mlir.global {:sym_name "g_buffer"
                       :linkage 10
                       :global_type !llvm.ptr
                       :constant false}
      (region
        (block []
          (def null_ptr (llvm.mlir.zero {:result !llvm.ptr}))
          (llvm.return null_ptr))))

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

;; Function that allocates and stores pointer in global (no return value)
(defn setup_buffer []
  ;; Allocate buffer
  (def size (: 64 i64))
  (def ptr (call !llvm.ptr malloc size))

  ;; Store in global
  (def g_buffer_addr (llvm.mlir.addressof {:global_name @g_buffer :result !llvm.ptr}))
  (llvm.store ptr g_buffer_addr)

  ;; Print
  (def ptr_int (llvm.ptrtoint {:result i64} ptr))
  (print "setup_buffer: ptr = 0x%lx\n" ptr_int)

  (func.return))

;; Function that reads file into the global buffer (no return value)
(defn do_fread []
  ;; Get buffer from global
  (def g_buffer_addr (llvm.mlir.addressof {:global_name @g_buffer :result !llvm.ptr}))
  (def ptr (llvm.load {:result !llvm.ptr} g_buffer_addr))

  ;; Print buffer pointer
  (def ptr_int (llvm.ptrtoint {:result i64} ptr))
  (print "do_fread: ptr = 0x%lx\n" ptr_int)

  ;; Open file
  (def path (llvm.mlir.addressof {:global_name @checkpoint_path :result !llvm.ptr}))
  (def mode (llvm.mlir.addressof {:global_name @read_mode_str :result !llvm.ptr}))
  (def file (call !llvm.ptr fopen path mode))

  ;; Read header (skip 1024 bytes)
  (def header_size (: 1024 i64))
  (def header_ptr (call !llvm.ptr malloc header_size))
  (def header_count (: 256 i64))
  (def four (: 4 i64))
  (def header_read (call i64 fread header_ptr four header_count file))

  ;; Read into buffer
  (def num_floats (: 16 i64))
  (def sizeof_f32 (: 4 i64))
  (def read_count (call i64 fread ptr sizeof_f32 num_floats file))
  (print "fread returned: %ld\n" read_count)

  ;; Check values in function
  (def read0 (llvm.load {:result f32} ptr))
  (def read0_64 (arith.extf {:result f64} read0))
  (def idx1 (: 1 i64))
  (def ptr1 (ptr-at f32 ptr idx1))
  (def read1 (llvm.load {:result f32} ptr1))
  (def read1_64 (arith.extf {:result f64} read1))
  (def z (: 0.0 f64))
  (print "In do_fread: val[0]=%f, val[1]=%f\n" read0_64 read1_64 z z)

  ;; Close file
  (def close_result (call i32 fclose file))

  (func.return))

(defn main [] -> i64
  ;; Set up buffer
  (call! setup_buffer)

  ;; Read file
  (call! do_fread)

  ;; Get buffer from global
  (def g_buffer_addr (llvm.mlir.addressof {:global_name @g_buffer :result !llvm.ptr}))
  (def ptr (llvm.load {:result !llvm.ptr} g_buffer_addr))

  ;; Print pointer
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
