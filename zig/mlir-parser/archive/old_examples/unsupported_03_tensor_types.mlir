// Example: Tensor types (shaped types)
// Grammar: tensor-type (part of builtin-type)

// Simple tensor constant
%0 = "arith.constant"() <{value = dense<[[1.0, 2.0], [3.0, 4.0]]> : tensor<2x2xf32>}> : () -> tensor<2x2xf32>

// Dynamic tensor shapes (? indicates unknown dimension)
%1 = "tensor.empty"() : () -> tensor<?x10xf64>

// Ranked tensor with various element types
%2 = "arith.constant"() <{value = dense<0> : tensor<4x8xi32>}> : () -> tensor<4x8xi32>
%3 = "arith.constant"() <{value = dense<0.0> : tensor<16xf16>}> : () -> tensor<16xf16>

// Unranked tensor (rank unknown at compile time)
%4 = "tensor.cast"(%0) : (tensor<2x2xf32>) -> tensor<*xf32>

// Multi-dimensional tensors
%5 = "arith.constant"() <{value = dense<0> : tensor<2x3x4x5xi64>}> : () -> tensor<2x3x4x5xi64>
