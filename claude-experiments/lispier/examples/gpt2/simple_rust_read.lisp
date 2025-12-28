;; Simple test using Rust to read the file and return pointer

(require-dialect func)
(require-dialect arith)
(require-dialect llvm)

(link-library :c)
(extern :gpt2-ffi)

(extern-fn printf (-> [!llvm.ptr] [i32]))
(extern-fn printf_1 (-> [!llvm.ptr i64] [i32]))
(extern-fn printf_4 (-> [!llvm.ptr f64 f64 f64 f64] [i32]))

(module
  (do
    (func.func {:sym_name "read_checkpoint_data"
                :function_type (-> [] [!llvm.ptr])
                :sym_visibility "private"})
    (func.func {:sym_name "get_test_data_ptr"
                :function_type (-> [] [!llvm.ptr])
                :sym_visibility "private"})
    (func.func {:sym_name "rust_load_f32"
                :function_type (-> [!llvm.ptr i64] [f32])
                :sym_visibility "private"})))

(defn main [] -> i64
  ;; Call Rust function to read checkpoint data
  (println "Calling read_checkpoint_data...")
  (def ptr (func.call {:result !llvm.ptr} "read_checkpoint_data"))

  ;; Print the returned pointer
  (def ptr_int (llvm.ptrtoint {:result i64} ptr))
  (print "Got ptr = 0x%lx\n" ptr_int)

  ;; Try 1: Read values using llvm.load
  (println "Reading with llvm.load...")
  (def val0 (llvm.load {:result f32} ptr))
  (def val0_64 (arith.extf {:result f64} val0))
  (def idx1 (: 1 i64))
  (def ptr1 (ptr-at f32 ptr idx1))
  (def val1 (llvm.load {:result f32} ptr1))
  (def val1_64 (arith.extf {:result f64} val1))
  (def z (: 0.0 f64))
  (print "llvm.load: [%f, %f]\n" val0_64 val1_64 z z)

  ;; Try 2: Read values using Rust function
  (println "Reading with rust_load_f32...")
  (def rval0 (func.call {:result f32} "rust_load_f32" ptr (: 0 i64)))
  (def rval0_64 (arith.extf {:result f64} rval0))
  (def rval1 (func.call {:result f32} "rust_load_f32" ptr (: 1 i64)))
  (def rval1_64 (arith.extf {:result f64} rval1))
  (print "rust_load_f32: [%f, %f]\n" rval0_64 rval1_64 z z)

  ;; Try 3: Get pointer from Rust global and try again
  (println "Getting ptr from Rust global...")
  (def ptr2 (func.call {:result !llvm.ptr} "get_test_data_ptr"))
  (def ptr2_int (llvm.ptrtoint {:result i64} ptr2))
  (print "Got ptr2 = 0x%lx\n" ptr2_int)

  (def val2_0 (llvm.load {:result f32} ptr2))
  (def val2_0_64 (arith.extf {:result f64} val2_0))
  (print "llvm.load from ptr2: %f\n" val2_0_64 z z z)

  (func.return (: 0 i64)))
