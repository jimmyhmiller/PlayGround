;; Minimal test with memcpy instead of fread

(require-dialect func)
(require-dialect arith)
(require-dialect llvm)
(require-dialect scf)

(link-library :c)

(extern-fn malloc (-> [i64] [!llvm.ptr]))
(extern-fn memcpy (-> [!llvm.ptr !llvm.ptr i64] [!llvm.ptr]))
(extern-fn printf (-> [!llvm.ptr] [i32]))
(extern-fn printf_1 (-> [!llvm.ptr i64] [i32]))
(extern-fn printf_4 (-> [!llvm.ptr f64 f64 f64 f64] [i32]))

(module
  (do))

;; Simple function that allocates, uses memcpy, returns pointer
(defn test_memcpy [] -> !llvm.ptr
  ;; Allocate source buffer with known values
  (def size (: 64 i64))
  (def src (call !llvm.ptr malloc size))

  ;; Write values to source
  (def val0 (: -123.456 f32))
  (llvm.store val0 src)
  (def idx1 (: 1 i64))
  (def src1 (ptr-at f32 src idx1))
  (def val1 (: -789.012 f32))
  (llvm.store val1 src1)

  ;; Allocate destination buffer
  (def dst (call !llvm.ptr malloc size))

  ;; Print pointers
  (def src_int (llvm.ptrtoint {:result i64} src))
  (def dst_int (llvm.ptrtoint {:result i64} dst))
  (print "src = 0x%lx\n" src_int)
  (print "dst = 0x%lx\n" dst_int)

  ;; Copy using memcpy
  (def copy_result (call !llvm.ptr memcpy dst src size))

  ;; Read back from dst to verify
  (def read0 (llvm.load {:result f32} dst))
  (def read0_64 (arith.extf {:result f64} read0))
  (def dst1 (ptr-at f32 dst idx1))
  (def read1 (llvm.load {:result f32} dst1))
  (def read1_64 (arith.extf {:result f64} read1))
  (def z (: 0.0 f64))
  (print "In function after memcpy: val[0]=%f, val[1]=%f\n" read0_64 read1_64 z z)

  ;; Return the destination pointer
  (func.return dst))

(defn main [] -> i64
  ;; Call the function and get the pointer
  (def ptr (call !llvm.ptr test_memcpy))

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
