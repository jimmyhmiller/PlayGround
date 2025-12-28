;; GPT-2 Checkpoint Loading
;;
;; Loads model weights from llm.c binary format (gpt2_124M.bin).
;;
;; Checkpoint Format (llm.c version 3):
;;   Header: 256 int32 values
;;     [0] magic = 20240326
;;     [1] version = 3
;;     [2] maxT = max sequence length (1024)
;;     [3] V = vocab size (50257)
;;     [4] L = number of layers (12)
;;     [5] NH = number of heads (12)
;;     [6] C = channels/hidden dim (768)
;;     [7] Vp = padded vocab size (50304)
;;
;;   Parameters: contiguous float32 array
;;     Order and sizes for GPT-2 Small (124M):
;;     1. wte: V × C = 50257 × 768 = 38,597,376 floats
;;     2. wpe: maxT × C = 1024 × 768 = 786,432 floats
;;     3. ln1w: L × C = 12 × 768 = 9,216 floats
;;     4. ln1b: L × C = 12 × 768 = 9,216 floats
;;     5. qkvw: L × 3C × C = 12 × 2304 × 768 = 21,233,664 floats
;;     6. qkvb: L × 3C = 12 × 2304 = 27,648 floats
;;     7. attprojw: L × C × C = 12 × 768 × 768 = 7,077,888 floats
;;     8. attprojb: L × C = 12 × 768 = 9,216 floats
;;     9. ln2w: L × C = 12 × 768 = 9,216 floats
;;     10. ln2b: L × C = 12 × 768 = 9,216 floats
;;     11. fcw: L × 4C × C = 12 × 3072 × 768 = 28,311,552 floats
;;     12. fcb: L × 4C = 12 × 3072 = 36,864 floats
;;     13. fcprojw: L × C × 4C = 12 × 768 × 3072 = 28,311,552 floats
;;     14. fcprojb: L × C = 12 × 768 = 9,216 floats
;;     15. lnfw: C = 768 floats
;;     16. lnfb: C = 768 floats
;;
;;   Total: ~124M parameters

;; Memory Layout for GPU:
;;
;; Each parameter tensor is loaded from file and transferred to GPU memory.
;; We use the async GPU pattern:
;;
;;   %host_wte = memref.alloc() : memref<50257x768xf32>
;;   ;; ... read from file into host_wte ...
;;   %gpu_wte, %t1 = gpu.alloc async [] () : memref<50257x768xf32>
;;   %t2 = gpu.memcpy async [%t1] %gpu_wte, %host_wte : memref<...>
;;
;; For inference, we keep all weights on GPU after initial transfer.

;; MLIR Pseudo-code for Loading:
;;
;; func.func @load_checkpoint(%path: !llvm.ptr)
;;     -> (!llvm.ptr, !llvm.ptr, !llvm.ptr, ...) {
;;   // Open file
;;   %file = llvm.call @fopen(%path, "rb") : ...
;;
;;   // Read header
;;   %header = memref.alloc() : memref<256xi32>
;;   llvm.call @fread(%header, 4, 256, %file)
;;
;;   // Verify magic and version
;;   %magic = memref.load %header[0] : memref<256xi32>
;;   // assert magic == 20240326
;;
;;   // Extract config
;;   %maxT = memref.load %header[2]
;;   %V = memref.load %header[3]
;;   %L = memref.load %header[4]
;;   %NH = memref.load %header[5]
;;   %C = memref.load %header[6]
;;
;;   // Allocate and read each parameter tensor
;;   %wte = memref.alloc(%V, %C) : memref<?x?xf32>
;;   llvm.call @fread(%wte, 4, %V*%C, %file)
;;
;;   // ... repeat for all 16 parameter tensors ...
;;
;;   llvm.call @fclose(%file)
;;   return %wte, %wpe, ...
;; }

;; Note: Full implementation requires LLVM dialect for file I/O.
;; For testing, we can use pre-allocated buffers with known values.

;; Test Configuration (for validation):
;; - Use smaller dimensions: B=1, T=4, C=16, L=2, NH=2
;; - Initialize weights with simple patterns (identity, zeros, ones)
;; - Compare outputs against CPU reference
