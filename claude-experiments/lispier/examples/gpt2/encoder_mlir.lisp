;; GPT-2 Encoder Step - MLIR Implementation
;;
;; This implements the GPT-2 encoder step (token + position embeddings) in MLIR.
;; The encoder converts input tokens to embeddings by:
;;   1. Looking up token embedding from wte (Word Token Embeddings)
;;   2. Adding position embedding from wpe (Word Position Embeddings)
;;
;; Parameters (from llm.c gpt2_124M):
;;   - vocab_size: 50257 (padded to 50304)
;;   - max_seq_len: 1024
;;   - channels: 768
;;
;; Run with: cargo run -- run examples/gpt2/encoder_mlir.lisp

(require-dialect func)
(require-dialect arith)
(require-dialect llvm)
(require-dialect scf)
(require-dialect memref)

;; Link C standard library for malloc, free, printf, file I/O
(link-library :c)

;; External C library functions
(extern-fn malloc (-> [i64] [!llvm.ptr]))
(extern-fn free (-> [!llvm.ptr] []))
(extern-fn fopen (-> [!llvm.ptr !llvm.ptr] [!llvm.ptr]))
(extern-fn fread (-> [!llvm.ptr i64 i64 !llvm.ptr] [i64]))
(extern-fn fclose (-> [!llvm.ptr] [i32]))

;; printf variants for print macro (N = number of format args)
(extern-fn printf (-> [!llvm.ptr] [i32]))
(extern-fn printf_1 (-> [!llvm.ptr i64] [i32]))
(extern-fn printf_4 (-> [!llvm.ptr f64 f64 f64 f64] [i32]))

(module
  (do
    ;; =========================================================================
    ;; Global variables for loaded checkpoint and debug state
    ;; =========================================================================

    ;; Params pointer (set by load_checkpoint)
    (llvm.mlir.global {:sym_name "g_params_ptr"
                       :linkage 10
                       :global_type !llvm.ptr
                       :constant false}
      (region
        (block []
          (def null_ptr (llvm.mlir.zero {:result !llvm.ptr}))
          (llvm.return null_ptr))))

    ;; Debug tokens pointer (set by load_debug_state)
    (llvm.mlir.global {:sym_name "g_debug_x_ptr"
                       :linkage 10
                       :global_type !llvm.ptr
                       :constant false}
      (region
        (block []
          (def null_ptr (llvm.mlir.zero {:result !llvm.ptr}))
          (llvm.return null_ptr))))

    ;; Debug sequence length
    (llvm.mlir.global {:sym_name "g_debug_seq_len"
                       :linkage 10
                       :global_type i64
                       :constant false}
      (region
        (block []
          (def zero (: 0 i64))
          (llvm.return zero))))


    ;; String constant for fopen mode "rb" with null terminator
    (llvm.mlir.global {:sym_name "read_mode_str"
                       :linkage 0
                       :global_type !llvm.array<3 x i8>
                       :constant true}
      (region
        (block []
          (def _str (llvm.mlir.constant {:value "rb\0" :result !llvm.array<3 x i8>}))
          (llvm.return _str))))

    ;; Global string constants for file paths
    (llvm.mlir.global {:sym_name "checkpoint_path"
                       :linkage 0
                       :global_type !llvm.array<39 x i8>
                       :constant true}
      (region
        (block []
          (def _str_val (llvm.mlir.constant {:value "/home/jimmyhmiller/llm.c/gpt2_124M.bin\0" :result !llvm.array<39 x i8>}))
          (llvm.return _str_val))))

    (llvm.mlir.global {:sym_name "debug_path"
                       :linkage 0
                       :global_type !llvm.array<51 x i8>
                       :constant true}
      (region
        (block []
          (def _str_val (llvm.mlir.constant {:value "/home/jimmyhmiller/llm.c/gpt2_124M_debug_state.bin\0" :result !llvm.array<51 x i8>}))
          (llvm.return _str_val))))))

;; =========================================================================
;; load_checkpoint - Load model parameters from llm.c checkpoint file
;;
;; File format:
;;   - 256 i32 header values (1024 bytes)
;;     - header[0]: magic = 20240326
;;     - header[1]: version = 3
;;     - header[2]: max_seq_len = 1024
;;     - header[3]: vocab_size = 50257
;;     - header[4]: num_layers = 12
;;     - header[5]: num_heads = 12
;;     - header[6]: channels = 768
;;     - header[7]: padded_vocab_size = 50304
;;   - Parameters as contiguous f32 array
;; =========================================================================
(defn load_checkpoint [(: path !llvm.ptr)] -> !llvm.ptr
  (println "Loading GPT-2 checkpoint...")

  ;; Open file
  (def read_mode (llvm.mlir.addressof {:global_name @read_mode_str :result !llvm.ptr}))
  (def file_ptr (call !llvm.ptr fopen path read_mode))

  ;; Allocate header buffer: 256 * 4 = 1024 bytes
  (def header_size (: 1024 i64))
  (def header_ptr (call !llvm.ptr malloc header_size))

  ;; Read header
  (def header_count (: 256 i64))
  (def four (: 4 i64))
  (def header_read (call i64 fread header_ptr four header_count file_ptr))

  ;; Extract config values from header
  (def c2 (: 2 i64))
  (def c6 (: 6 i64))
  (def c7 (: 7 i64))

  (def h2_ptr (ptr-at i32 header_ptr c2))
  (def max_seq_len_i32 (llvm.load {:result i32} h2_ptr))
  (def max_seq_len (arith.extsi {:result i64} max_seq_len_i32))

  (def h6_ptr (ptr-at i32 header_ptr c6))
  (def channels_i32 (llvm.load {:result i32} h6_ptr))
  (def channels (arith.extsi {:result i64} channels_i32))

  (def h7_ptr (ptr-at i32 header_ptr c7))
  (def padded_vocab_size_i32 (llvm.load {:result i32} h7_ptr))
  (def padded_vocab_size (arith.extsi {:result i64} padded_vocab_size_i32))

  ;; Calculate number of parameters (simplified for GPT-2 124M)
  (def num_params (: 124439808 i64))

  ;; Allocate params buffer: num_params * 4 bytes
  (def sizeof_f32 (: 4 i64))
  (def params_bytes (arith.muli num_params sizeof_f32))
  (def params_ptr (call !llvm.ptr malloc params_bytes))

  ;; Debug: print pointer value from malloc
  (def params_ptr_int (llvm.ptrtoint {:result i64} params_ptr))
  (print "malloc returned ptr: 0x%lx\n" params_ptr_int)

  ;; Skip fread for now - just write test values manually to debug memory issue
  (def test_val (: -999.0 f32))
  (llvm.store test_val params_ptr)
  (def idx1 (: 1 i64))
  (def ptr1 (ptr-at f32 params_ptr idx1))
  (def test_val2 (: -888.0 f32))
  (llvm.store test_val2 ptr1)

  ;; Verify they were written
  (def check_val (llvm.load {:result f32} params_ptr))
  (def check_val_64 (arith.extf {:result f64} check_val))
  (print "After manual store: val[0] = %f\n" check_val_64 check_val_64 check_val_64 check_val_64)

  ;; Actually call fread to load the real data
  (def params_read (call i64 fread params_ptr sizeof_f32 num_params file_ptr))

  ;; Debug: check how many were actually read
  (print "fread returned: %ld\n" params_read)

  ;; Debug: check values immediately after fread (before storing global)
  (def idx0 (: 0 i64))
  (def idx1 (: 1 i64))
  (def test_ptr0 (ptr-at f32 params_ptr idx0))
  (def test_val0 (llvm.load {:result f32} test_ptr0))
  (def test_val0_64 (arith.extf {:result f64} test_val0))
  (def test_ptr1 (ptr-at f32 params_ptr idx1))
  (def test_val1 (llvm.load {:result f32} test_ptr1))
  (def test_val1_64 (arith.extf {:result f64} test_val1))
  (def zero_64 (: 0.0 f64))
  (print "[load_checkpoint] wte[0]=%f, wte[1]=%f\n" test_val0_64 test_val1_64 zero_64 zero_64)

  ;; Store params pointer in global
  (def g_params_ptr_addr (llvm.mlir.addressof {:global_name @g_params_ptr :result !llvm.ptr}))
  (def params_ptr_int2 (llvm.ptrtoint {:result i64} params_ptr))
  (print "storing to global ptr: 0x%lx\n" params_ptr_int2)
  (llvm.store params_ptr g_params_ptr_addr)

  ;; Print loaded params info
  (print "Loaded %ld parameters\n" num_params)

  ;; Close file and free header
  (def close_result (call i32 fclose file_ptr))
  (call! free header_ptr)

  ;; Return the params pointer directly
  (func.return params_ptr))

;; =========================================================================
;; load_debug_state - Load debug tokens and expected outputs
;; =========================================================================
(defn load_debug_state [(: path !llvm.ptr)] -> i32
  ;; Open file
  (def read_mode (llvm.mlir.addressof {:global_name @read_mode_str :result !llvm.ptr}))
  (def file_ptr (call !llvm.ptr fopen path read_mode))

  ;; Allocate header buffer: 256 * 4 = 1024 bytes
  (def header_size (: 1024 i64))
  (def header_ptr (call !llvm.ptr malloc header_size))

  ;; Read header
  (def header_count (: 256 i64))
  (def four (: 4 i64))
  (def header_read (call i64 fread header_ptr four header_count file_ptr))

  ;; Extract batch_size and seq_len from header
  (def c2 (: 2 i64))
  (def c3 (: 3 i64))

  (def h2_ptr (ptr-at i32 header_ptr c2))
  (def batch_size_i32 (llvm.load {:result i32} h2_ptr))
  (def batch_size (arith.extsi {:result i64} batch_size_i32))

  (def h3_ptr (ptr-at i32 header_ptr c3))
  (def seq_len_i32 (llvm.load {:result i32} h3_ptr))
  (def seq_len (arith.extsi {:result i64} seq_len_i32))

  ;; Calculate total tokens: batch_size * seq_len
  (def total_tokens (arith.muli batch_size seq_len))

  ;; Allocate tokens buffer: total_tokens * 4 bytes
  (def sizeof_i32 (: 4 i64))
  (def tokens_bytes (arith.muli total_tokens sizeof_i32))
  (def tokens_ptr (call !llvm.ptr malloc tokens_bytes))

  ;; Read input tokens
  (def tokens_read (call i64 fread tokens_ptr sizeof_i32 total_tokens file_ptr))

  ;; Store tokens pointer in global
  (def g_debug_x_ptr_addr (llvm.mlir.addressof {:global_name @g_debug_x_ptr :result !llvm.ptr}))
  (llvm.store tokens_ptr g_debug_x_ptr_addr)

  ;; Store seq_len in global (we just use first sequence for now)
  (def g_debug_seq_len_addr (llvm.mlir.addressof {:global_name @g_debug_seq_len :result !llvm.ptr}))
  (llvm.store seq_len g_debug_seq_len_addr)

  ;; Print loaded debug info
  (print "Loaded debug state: seq_len=%ld\n" seq_len)

  ;; Close file and free header
  (def close_result (call i32 fclose file_ptr))
  (call! free header_ptr)

  ;; Return success
  (func.return (: 0 i32)))

;; =========================================================================
;; Encoder forward pass - implemented in MLIR
;; Takes:
;;   params_ptr: pointer to model parameters (wte is at offset 0, wpe follows)
;;   tokens_ptr: pointer to input token ids (i32)
;;   seq_len: sequence length
;;   output_ptr: pointer to output buffer
;;
;; wte layout: [vocab_size=50304, channels=768]
;; wpe layout: [max_seq_len=1024, channels=768]
;; output layout: [seq_len, channels=768]
;; =========================================================================
(defn encoder_forward [(: params_ptr !llvm.ptr) (: tokens_ptr !llvm.ptr) (: seq_len i64) (: output_ptr !llvm.ptr)]
  ;; Constants
  (def channels (: 768 i64))
  (def vocab_size_padded (: 50304 i64))

  ;; wte is at params_ptr offset 0
  ;; wpe is at params_ptr offset (vocab_size_padded * channels)
  (def wpe_offset (arith.muli vocab_size_padded channels))

  ;; wpe_ptr = params_ptr + wpe_offset (in f32 elements)
  (def wpe_ptr (ptr-at f32 params_ptr wpe_offset))

  ;; Loop over each position in the sequence
  (def c0 (: 0 i64))
  (def c1 (: 1 i64))
  (scf.for c0 seq_len c1
    (region
      (block [(: pos i64)]
        ;; Get token id at this position
        (def token_ptr (ptr-at i32 tokens_ptr pos))
        (def token_i32 (llvm.load {:result i32} token_ptr))
        (def token (arith.extsi {:result i64} token_i32))

        ;; Calculate offset into wte: token * channels
        (def wte_token_offset (arith.muli token channels))

        ;; Calculate offset into wpe: pos * channels
        (def wpe_pos_offset (arith.muli pos channels))

        ;; Calculate output offset: pos * channels
        (def out_offset (arith.muli pos channels))

        ;; Loop over each channel and add embeddings
        (scf.for c0 channels c1
          (region
            (block [(: c i64)]
              ;; Load wte[token, c]
              (def wte_idx (arith.addi wte_token_offset c))
              (def wte_elem_ptr (ptr-at f32 params_ptr wte_idx))
              (def wte_val (llvm.load {:result f32} wte_elem_ptr))

              ;; Load wpe[pos, c]
              (def wpe_idx (arith.addi wpe_pos_offset c))
              (def wpe_elem_ptr (ptr-at f32 wpe_ptr wpe_idx))
              (def wpe_val (llvm.load {:result f32} wpe_elem_ptr))

              ;; Add embeddings
              (def sum (arith.addf wte_val wpe_val))

              ;; Store to output
              (def out_idx (arith.addi out_offset c))
              (def out_elem_ptr (ptr-at f32 output_ptr out_idx))
              (llvm.store sum out_elem_ptr)

              (scf.yield))))

        (scf.yield))))

  (func.return))

;; =========================================================================
;; Main function - load checkpoint, run encoder, validate
;; =========================================================================
(defn main [] -> i64
  ;; Load checkpoint (our lisp function) - now returns pointer directly
  (def checkpoint_path (llvm.mlir.addressof {:global_name @checkpoint_path :result !llvm.ptr}))
  (def params_ptr (call !llvm.ptr load_checkpoint checkpoint_path))

  ;; Check params immediately after load_checkpoint returns
  (def params_ptr_int (llvm.ptrtoint {:result i64} params_ptr))
  (print "Got params_ptr from load_checkpoint: 0x%lx\n" params_ptr_int)

  (def wte0_early (llvm.load {:result f32} params_ptr))
  (def wte0_early_64 (arith.extf {:result f64} wte0_early))
  (print "Right after load_checkpoint return: wte[0]=%f\n" wte0_early_64 wte0_early_64 wte0_early_64 wte0_early_64)

  ;; Load debug state (our lisp function)
  (def debug_path (llvm.mlir.addressof {:global_name @debug_path :result !llvm.ptr}))
  (def debug_result (call i32 load_debug_state debug_path))

  ;; Debug: check if params_ptr is null
  (def null_ptr (llvm.mlir.zero {:result !llvm.ptr}))
  (def is_null (llvm.icmp {:predicate 0} params_ptr null_ptr))
  (scf.if is_null
    (region
      (block []
        (println "ERROR: params_ptr is NULL!")
        (scf.yield)))
    (region
      (block []
        (println "params_ptr is valid")
        (scf.yield))))

  (def g_debug_x_ptr_addr (llvm.mlir.addressof {:global_name @g_debug_x_ptr :result !llvm.ptr}))
  (def tokens_ptr (llvm.load {:result !llvm.ptr} g_debug_x_ptr_addr))

  (def g_debug_seq_len_addr (llvm.mlir.addressof {:global_name @g_debug_seq_len :result !llvm.ptr}))
  (def seq_len (llvm.load {:result i64} g_debug_seq_len_addr))

  ;; Allocate output buffer: seq_len * 768 * sizeof(f32)
  (def channels (: 768 i64))
  (def sizeof_f32 (: 4 i64))
  (def output_elems (arith.muli seq_len channels))
  (def output_bytes (arith.muli output_elems sizeof_f32))
  (def output_ptr (call !llvm.ptr malloc output_bytes))

  ;; Debug: print first token
  (def tok0_ptr (ptr-at i32 tokens_ptr (: 0 i64)))
  (def tok0_i32 (llvm.load {:result i32} tok0_ptr))
  (def tok0_i64 (arith.extsi {:result i64} tok0_i32))
  (print "First token: %ld\n" tok0_i64)

  ;; Debug: Check first two f32 values in params
  (def idx0 (: 0 i64))
  (def idx1 (: 1 i64))
  (def wte0_ptr (ptr-at f32 params_ptr idx0))
  (def wte0 (llvm.load {:result f32} wte0_ptr))
  (def wte0_64 (arith.extf {:result f64} wte0))

  (def wte1_ptr (ptr-at f32 params_ptr idx1))
  (def wte1 (llvm.load {:result f32} wte1_ptr))
  (def wte1_64 (arith.extf {:result f64} wte1))

  ;; Print both
  (def zero_64 (: 0.0 f64))
  (print "wte[0]=%f, wte[1]=%f\n" wte0_64 wte1_64 zero_64 zero_64)

  ;; Run encoder (our MLIR implementation!)
  (call! encoder_forward params_ptr tokens_ptr seq_len output_ptr)

  ;; Load first few output values for verification
  (def c0 (: 0 i64))
  (def c1 (: 1 i64))
  (def c2 (: 2 i64))
  (def c3 (: 3 i64))

  (def out0_ptr (ptr-at f32 output_ptr c0))
  (def out0 (llvm.load {:result f32} out0_ptr))
  (def out1_ptr (ptr-at f32 output_ptr c1))
  (def out1 (llvm.load {:result f32} out1_ptr))
  (def out2_ptr (ptr-at f32 output_ptr c2))
  (def out2 (llvm.load {:result f32} out2_ptr))
  (def out3_ptr (ptr-at f32 output_ptr c3))
  (def out3 (llvm.load {:result f32} out3_ptr))

  ;; Print encoder output - need to convert f32 to f64 for printf
  (def out0_64 (arith.extf {:result f64} out0))
  (def out1_64 (arith.extf {:result f64} out1))
  (def out2_64 (arith.extf {:result f64} out2))
  (def out3_64 (arith.extf {:result f64} out3))
  (print "Encoder output: [%f, %f, %f, %f, ...]\n" out0_64 out1_64 out2_64 out3_64)

  ;; Free output
  (call! free output_ptr)

  ;; Return 0 for success
  (func.return (: 0 i64)))