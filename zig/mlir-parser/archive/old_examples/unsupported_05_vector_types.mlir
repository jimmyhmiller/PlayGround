// Example: Vector types (SIMD vectors)
// Grammar: vector-type (part of builtin-type)

// Simple vector constant
%0 = "arith.constant"() <{value = dense<[1.0, 2.0, 3.0, 4.0]> : vector<4xf32>}> : () -> vector<4xf32>

// Multi-dimensional vector
%1 = "arith.constant"() <{value = dense<0.0> : vector<4x8xf64>}> : () -> vector<4x8xf64>

// Vector operations
%2 = "arith.constant"() <{value = dense<[5.0, 6.0, 7.0, 8.0]> : vector<4xf32>}> : () -> vector<4xf32>
%3 = "arith.addf"(%0, %2) : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>
%4 = "arith.mulf"(%0, %2) : (vector<4xf32>, vector<4xf32>) -> vector<4xf32>

// Scalable vectors (for SVE, RVV, etc.)
%5 = "arith.constant"() <{value = dense<0> : vector<[4]xi32>}> : () -> vector<[4]xi32>

// Vector with different element types
%6 = "arith.constant"() <{value = dense<0> : vector<8xi64>}> : () -> vector<8xi64>
%7 = "arith.constant"() <{value = dense<0> : vector<16xi8>}> : () -> vector<16xi8>
