;; Minimal test to debug memory persistence across function calls

(require-dialect func)
(require-dialect arith)
(require-dialect llvm)
(require-dialect scf)

(link-library :c)

(extern-fn malloc (-> [i64] [!llvm.ptr]))
(extern-fn printf (-> [!llvm.ptr] [i32]))
(extern-fn printf_1 (-> [!llvm.ptr i64] [i32]))
(extern-fn printf_4 (-> [!llvm.ptr f64 f64 f64 f64] [i32]))

(module
  (do))

;; Simple function that allocates, writes, and returns pointer
(defn test_alloc_and_write [] -> !llvm.ptr
  ;; Allocate 16 floats
  (def size (: 64 i64))
  (def ptr (call !llvm.ptr malloc size))

  ;; Print the pointer
  (def ptr_int (llvm.ptrtoint {:result i64} ptr))
  (print "In test_alloc_and_write: ptr = 0x%lx\n" ptr_int)

  ;; Write known values
  (def val0 (: -123.456 f32))
  (llvm.store val0 ptr)

  (def idx1 (: 1 i64))
  (def ptr1 (ptr-at f32 ptr idx1))
  (def val1 (: -789.012 f32))
  (llvm.store val1 ptr1)

  ;; Read back to verify
  (def read0 (llvm.load {:result f32} ptr))
  (def read0_64 (arith.extf {:result f64} read0))
  (def read1 (llvm.load {:result f32} ptr1))
  (def read1_64 (arith.extf {:result f64} read1))
  (def z (: 0.0 f64))
  (print "In function: val[0]=%f, val[1]=%f\n" read0_64 read1_64 z z)

  ;; Return the pointer
  (func.return ptr))

(defn main [] -> i64
  ;; Call the function and get the pointer
  (def ptr (call !llvm.ptr test_alloc_and_write))

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
