// Example: Memref types (memory references)
// Grammar: memref-type (part of builtin-type)

// Basic memref allocation
%0 = "memref.alloc"() : () -> memref<16x16xf64>

// Memref with affine map (memory layout)
%1 = "memref.alloc"() : () -> memref<10x20xf32, affine_map<(d0, d1) -> (d0 * 20 + d1)>>

// Dynamic memref (sizes determined at runtime)
%c10 = "arith.constant"() <{value = 10 : index}> : () -> index
%c20 = "arith.constant"() <{value = 20 : index}> : () -> index
%2 = "memref.alloc"(%c10, %c20) : (index, index) -> memref<?x?xf32>

// Memref load and store
%idx0 = "arith.constant"() <{value = 0 : index}> : () -> index
%idx1 = "arith.constant"() <{value = 1 : index}> : () -> index
%val = "memref.load"(%0, %idx0, %idx1) : (memref<16x16xf64>, index, index) -> f64
"memref.store"(%val, %0, %idx1, %idx0) : (f64, memref<16x16xf64>, index, index) -> ()

// Memref with memory space (e.g., GPU memory)
%3 = "memref.alloc"() : () -> memref<256xf32, 1>

// Unranked memref
%4 = "memref.cast"(%0) : (memref<16x16xf64>) -> memref<*xf64>
