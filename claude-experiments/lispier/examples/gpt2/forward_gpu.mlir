// Simplified GPT-2 Forward Pass - Single Layer
// B=1, T=4, C=16, NH=2, hs=8
// Testing the operation flow before scaling up

module attributes {gpu.container_module} {
  // GPU Kernels
  gpu.module @kernels {
    // LayerNorm kernel - one thread per (b,t) computes normalization over C
    gpu.func @layernorm(%inp: memref<1x4x16xf32>, %out: memref<1x4x16xf32>,
                        %gamma: memref<16xf32>, %beta: memref<16xf32>) kernel {
      %t = gpu.thread_id x  // time position
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      %eps = arith.constant 1.0e-5 : f32
      %zero = arith.constant 0.0 : f32
      %c16f = arith.constant 16.0 : f32
      
      // Compute mean
      %sum = scf.for %c = %c0 to %c16 step %c1 iter_args(%acc = %zero) -> f32 {
        %v = memref.load %inp[%c0, %t, %c] : memref<1x4x16xf32>
        %new_acc = arith.addf %acc, %v : f32
        scf.yield %new_acc : f32
      }
      %mean = arith.divf %sum, %c16f : f32
      
      // Compute variance
      %var_sum = scf.for %c = %c0 to %c16 step %c1 iter_args(%acc = %zero) -> f32 {
        %v = memref.load %inp[%c0, %t, %c] : memref<1x4x16xf32>
        %diff = arith.subf %v, %mean : f32
        %sq = arith.mulf %diff, %diff : f32
        %new_acc = arith.addf %acc, %sq : f32
        scf.yield %new_acc : f32
      }
      %var = arith.divf %var_sum, %c16f : f32
      %var_eps = arith.addf %var, %eps : f32
      %rstd = math.rsqrt %var_eps : f32
      
      // Normalize and scale
      scf.for %c = %c0 to %c16 step %c1 {
        %v = memref.load %inp[%c0, %t, %c] : memref<1x4x16xf32>
        %centered = arith.subf %v, %mean : f32
        %normed = arith.mulf %centered, %rstd : f32
        %g = memref.load %gamma[%c] : memref<16xf32>
        %b = memref.load %beta[%c] : memref<16xf32>
        %scaled = arith.mulf %normed, %g : f32
        %result = arith.addf %scaled, %b : f32
        memref.store %result, %out[%c0, %t, %c] : memref<1x4x16xf32>
      }
      
      gpu.return
    }
    
    // GELU kernel - element-wise
    gpu.func @gelu(%inp: memref<1x4x64xf32>, %out: memref<1x4x64xf32>) kernel {
      %tx = gpu.thread_id x
      %ty = gpu.thread_id y
      %c0 = arith.constant 0 : index
      
      // tanh approximation: gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
      %sqrt_2_over_pi = arith.constant 0.7978845608 : f32
      %coef = arith.constant 0.044715 : f32
      %half = arith.constant 0.5 : f32
      %one = arith.constant 1.0 : f32
      
      %x = memref.load %inp[%c0, %tx, %ty] : memref<1x4x64xf32>
      %x2 = arith.mulf %x, %x : f32
      %x3 = arith.mulf %x2, %x : f32
      %coef_x3 = arith.mulf %coef, %x3 : f32
      %inner = arith.addf %x, %coef_x3 : f32
      %scaled = arith.mulf %sqrt_2_over_pi, %inner : f32
      %tanh_val = math.tanh %scaled : f32
      %one_plus = arith.addf %one, %tanh_val : f32
      %half_x = arith.mulf %half, %x : f32
      %result = arith.mulf %half_x, %one_plus : f32
      memref.store %result, %out[%c0, %tx, %ty] : memref<1x4x64xf32>
      
      gpu.return
    }
    
    // Residual add kernel
    gpu.func @residual_add(%out: memref<1x4x16xf32>, 
                           %a: memref<1x4x16xf32>, 
                           %b: memref<1x4x16xf32>) kernel {
      %tx = gpu.thread_id x  // time
      %ty = gpu.thread_id y  // channel
      %c0 = arith.constant 0 : index
      
      %va = memref.load %a[%c0, %tx, %ty] : memref<1x4x16xf32>
      %vb = memref.load %b[%c0, %tx, %ty] : memref<1x4x16xf32>
      %sum = arith.addf %va, %vb : f32
      memref.store %sum, %out[%c0, %tx, %ty] : memref<1x4x16xf32>
      
      gpu.return
    }
  }
  
  // Test the forward pass components
  func.func @main() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %c16 = arith.constant 16 : index
    %c64 = arith.constant 64 : index
    %one = arith.constant 1.0 : f32
    %zero = arith.constant 0.0 : f32
    
    // Allocate host tensors
    %inp_host = memref.alloc() : memref<1x4x16xf32>
    %out_host = memref.alloc() : memref<1x4x16xf32>
    %gamma_host = memref.alloc() : memref<16xf32>
    %beta_host = memref.alloc() : memref<16xf32>
    
    // Initialize: input = 1.0, gamma = 1.0, beta = 0.0
    scf.for %t = %c0 to %c4 step %c1 {
      scf.for %c = %c0 to %c16 step %c1 {
        memref.store %one, %inp_host[%c0, %t, %c] : memref<1x4x16xf32>
      }
    }
    scf.for %c = %c0 to %c16 step %c1 {
      memref.store %one, %gamma_host[%c] : memref<16xf32>
      memref.store %zero, %beta_host[%c] : memref<16xf32>
    }
    
    // GPU allocations
    %inp_gpu, %t1 = gpu.alloc async [] () : memref<1x4x16xf32>
    %out_gpu, %t2 = gpu.alloc async [%t1] () : memref<1x4x16xf32>
    %gamma_gpu, %t3 = gpu.alloc async [%t2] () : memref<16xf32>
    %beta_gpu, %t4 = gpu.alloc async [%t3] () : memref<16xf32>
    
    // Copy to GPU
    %t5 = gpu.memcpy async [%t4] %inp_gpu, %inp_host : memref<1x4x16xf32>, memref<1x4x16xf32>
    %t6 = gpu.memcpy async [%t5] %gamma_gpu, %gamma_host : memref<16xf32>, memref<16xf32>
    %t7 = gpu.memcpy async [%t6] %beta_gpu, %beta_host : memref<16xf32>, memref<16xf32>
    
    // Launch layernorm: 1 block, 4 threads (one per time position)
    %t8 = gpu.launch_func async [%t7] @kernels::@layernorm
      blocks in (%c1, %c1, %c1) threads in (%c4, %c1, %c1)
      args(%inp_gpu : memref<1x4x16xf32>, %out_gpu : memref<1x4x16xf32>,
           %gamma_gpu : memref<16xf32>, %beta_gpu : memref<16xf32>)
    
    // Copy back
    %t9 = gpu.memcpy async [%t8] %out_host, %out_gpu : memref<1x4x16xf32>, memref<1x4x16xf32>
    gpu.wait [%t9]
    
    // Print result - layernorm of constant 1 should be 0 (centered at mean)
    %v00 = memref.load %out_host[%c0, %c0, %c0] : memref<1x4x16xf32>
    vector.print %v00 : f32
    
    // Cleanup
    gpu.dealloc %inp_gpu : memref<1x4x16xf32>
    gpu.dealloc %out_gpu : memref<1x4x16xf32>
    gpu.dealloc %gamma_gpu : memref<16xf32>
    gpu.dealloc %beta_gpu : memref<16xf32>
    memref.dealloc %inp_host : memref<1x4x16xf32>
    memref.dealloc %out_host : memref<1x4x16xf32>
    memref.dealloc %gamma_host : memref<16xf32>
    memref.dealloc %beta_host : memref<16xf32>
    
    return
  }
}
