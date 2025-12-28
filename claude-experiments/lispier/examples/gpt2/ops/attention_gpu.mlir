// Simplified Attention Implementation for GPU
// B=1, T=4, C=64, NH=4, hs=16
// Input: qkv (B,T,3*C) where Q,K,V are interleaved
// Output: out (B,T,C)

module attributes {gpu.container_module} {
  // Single-head attention kernel for position t
  // Computes attention output for one (batch, time, head) position
  gpu.module @kernels {
    // Kernel: compute attention for one head at one position
    // Each block handles one (b,t,h) tuple
    gpu.func @attention_head(
      %qkv: memref<1x4x192xf32>,  // (B,T,3*C) input with Q,K,V
      %out: memref<1x4x64xf32>,   // (B,T,C) output
      %preatt: memref<1x4x4x4xf32>, // (B,NH,T,T) pre-softmax scores
      %att: memref<1x4x4x4xf32>     // (B,NH,T,T) post-softmax attention
    ) kernel {
      // Thread indices: tx = head position within head, ty = time position
      %tx = gpu.thread_id x
      %ty = gpu.thread_id y
      %bx = gpu.block_id x
      
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %c16 = arith.constant 16 : index
      %c64 = arith.constant 64 : index
      %c128 = arith.constant 128 : index
      %hs = arith.constant 16 : index
      %zero = arith.constant 0.0 : f32
      %neg_inf = arith.constant -1.0e9 : f32
      %scale = arith.constant 0.25 : f32  // 1/sqrt(16)
      
      // Only first thread in x does the work for now (simplified)
      %is_first = arith.cmpi eq, %tx, %c0 : index
      scf.if %is_first {
        // For time position ty, head bx
        // Query offset: b*T*3C + t*3C + h*hs = 0 + ty*192 + bx*16
        // Key offset: b*T*3C + t2*3C + h*hs + C = ty*192 + bx*16 + 64
        // Value offset: b*T*3C + t2*3C + h*hs + 2*C = ty*192 + bx*16 + 128
        
        // Compute attention scores for this (b,t,h)
        %max_val = arith.constant -1.0e9 : f32
        
        // Pass 1: Compute Q dot K for t2 <= ty
        %max_result = scf.for %t2 = %c0 to %c4 step %c1 iter_args(%maxv = %max_val) -> f32 {
          %is_valid = arith.cmpi sle, %t2, %ty : index
          %score_or_neginf = scf.if %is_valid -> f32 {
            // Compute Q[t] dot K[t2]
            %dot = scf.for %i = %c0 to %hs step %c1 iter_args(%acc = %zero) -> f32 {
              // Query: qkv[0, ty, bx*hs + i]
              %q_offset = arith.muli %bx, %hs : index
              %q_idx = arith.addi %q_offset, %i : index
              %q_val = memref.load %qkv[%c0, %ty, %q_idx] : memref<1x4x192xf32>
              
              // Key: qkv[0, t2, bx*hs + i + C]
              %k_offset_base = arith.addi %q_idx, %c64 : index
              %k_val = memref.load %qkv[%c0, %t2, %k_offset_base] : memref<1x4x192xf32>
              
              %prod = arith.mulf %q_val, %k_val : f32
              %new_acc = arith.addf %acc, %prod : f32
              scf.yield %new_acc : f32
            }
            // Scale by 1/sqrt(hs)
            %scaled = arith.mulf %dot, %scale : f32
            scf.yield %scaled : f32
          } else {
            scf.yield %neg_inf : f32
          }
          
          // Store in preatt
          memref.store %score_or_neginf, %preatt[%c0, %bx, %ty, %t2] : memref<1x4x4x4xf32>
          
          // Update max
          %cmp = arith.cmpf ogt, %score_or_neginf, %maxv : f32
          %new_max = arith.select %cmp, %score_or_neginf, %maxv : f32
          scf.yield %new_max : f32
        }
        
        // Pass 2: Compute exp and sum
        %expsum = scf.for %t2 = %c0 to %c4 step %c1 iter_args(%sum = %zero) -> f32 {
          %score = memref.load %preatt[%c0, %bx, %ty, %t2] : memref<1x4x4x4xf32>
          %shifted = arith.subf %score, %max_result : f32
          %expv = math.exp %shifted : f32
          memref.store %expv, %att[%c0, %bx, %ty, %t2] : memref<1x4x4x4xf32>
          %new_sum = arith.addf %sum, %expv : f32
          scf.yield %new_sum : f32
        }
        
        // Pass 3: Normalize
        scf.for %t2 = %c0 to %c4 step %c1 {
          %expv = memref.load %att[%c0, %bx, %ty, %t2] : memref<1x4x4x4xf32>
          %normalized = arith.divf %expv, %expsum : f32
          memref.store %normalized, %att[%c0, %bx, %ty, %t2] : memref<1x4x4x4xf32>
        }
        
        // Pass 4: Weighted sum of values
        scf.for %i = %c0 to %hs step %c1 {
          %acc = scf.for %t2 = %c0 to %c4 step %c1 iter_args(%a = %zero) -> f32 {
            %att_weight = memref.load %att[%c0, %bx, %ty, %t2] : memref<1x4x4x4xf32>
            
            // Value: qkv[0, t2, bx*hs + i + 2*C]
            %v_offset_base = arith.muli %bx, %hs : index
            %v_idx_1 = arith.addi %v_offset_base, %i : index
            %v_idx = arith.addi %v_idx_1, %c128 : index
            %v_val = memref.load %qkv[%c0, %t2, %v_idx] : memref<1x4x192xf32>
            
            %prod = arith.mulf %att_weight, %v_val : f32
            %new_a = arith.addf %a, %prod : f32
            scf.yield %new_a : f32
          }
          
          // Store output: out[0, ty, bx*hs + i]
          %out_offset = arith.muli %bx, %hs : index
          %out_idx = arith.addi %out_offset, %i : index
          memref.store %acc, %out[%c0, %ty, %out_idx] : memref<1x4x64xf32>
        }
      }
      
      gpu.return
    }
  }
  
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    
    // Host memory
    %qkv_host = memref.alloc() : memref<1x4x192xf32>
    %out_host = memref.alloc() : memref<1x4x64xf32>
    
    // Initialize QKV with simple values
    // Q = 0.1, K = 0.1, V = 1.0
    %q_val = arith.constant 0.1 : f32
    %k_val = arith.constant 0.1 : f32  
    %v_val = arith.constant 1.0 : f32
    
    scf.for %t = %c0 to %c4 step %c1 {
      scf.for %c = %c0 to %c16 step %c1 {
        // Set all 4 heads
        scf.for %h = %c0 to %c4 step %c1 {
          %offset = arith.muli %h, %c16 : index
          %idx = arith.addi %offset, %c : index
          memref.store %q_val, %qkv_host[%c0, %t, %idx] : memref<1x4x192xf32>
          
          %k_offset = arith.addi %idx, %c16 : index
          %k_offset2 = arith.muli %k_offset, %c1 : index  // hack to avoid constant fold
          %c64_idx = arith.constant 64 : index
          %k_idx = arith.addi %idx, %c64_idx : index
          memref.store %k_val, %qkv_host[%c0, %t, %k_idx] : memref<1x4x192xf32>
          
          %c128_idx = arith.constant 128 : index
          %v_idx = arith.addi %idx, %c128_idx : index
          memref.store %v_val, %qkv_host[%c0, %t, %v_idx] : memref<1x4x192xf32>
        }
      }
    }
    
    // GPU memory
    %qkv_gpu, %t1 = gpu.alloc async [] () : memref<1x4x192xf32>
    %out_gpu, %t2 = gpu.alloc async [%t1] () : memref<1x4x64xf32>
    %preatt_gpu, %t3 = gpu.alloc async [%t2] () : memref<1x4x4x4xf32>
    %att_gpu, %t4 = gpu.alloc async [%t3] () : memref<1x4x4x4xf32>
    
    // Copy to GPU
    %t5 = gpu.memcpy async [%t4] %qkv_gpu, %qkv_host : memref<1x4x192xf32>, memref<1x4x192xf32>
    
    // Launch kernel: 4 blocks (one per head), 4 threads per block (one per time)
    %t6 = gpu.launch_func async [%t5] @kernels::@attention_head 
      blocks in (%c4, %c1, %c1) threads in (%c1, %c4, %c1)
      args(%qkv_gpu : memref<1x4x192xf32>, 
           %out_gpu : memref<1x4x64xf32>,
           %preatt_gpu : memref<1x4x4x4xf32>,
           %att_gpu : memref<1x4x4x4xf32>)
    
    // Copy back
    %t7 = gpu.memcpy async [%t6] %out_host, %out_gpu : memref<1x4x64xf32>, memref<1x4x64xf32>
    gpu.wait [%t7]
    
    // Print output
    %out00 = memref.load %out_host[%c0, %c0, %c0] : memref<1x4x64xf32>
    %out10 = memref.load %out_host[%c0, %c1, %c0] : memref<1x4x64xf32>
    vector.print %out00 : f32
    vector.print %out10 : f32
    
    // Cleanup
    gpu.dealloc %qkv_gpu : memref<1x4x192xf32>
    gpu.dealloc %out_gpu : memref<1x4x64xf32>
    gpu.dealloc %preatt_gpu : memref<1x4x4x4xf32>
    gpu.dealloc %att_gpu : memref<1x4x4x4xf32>
    memref.dealloc %qkv_host : memref<1x4x192xf32>
    memref.dealloc %out_host : memref<1x4x64xf32>
    
    return
  }
}
