// Example: Complex and tuple types
// Grammar: complex-type, tuple-type (part of builtin-type)

// Complex number type
%0 = "arith.constant"() <{value = (1.0, 2.0) : complex<f32>}> : () -> complex<f32>
%1 = "arith.constant"() <{value = (3.0, 4.0) : complex<f64>}> : () -> complex<f64>

// Complex operations
%2 = "complex.add"(%0, %0) : (complex<f32>, complex<f32>) -> complex<f32>
%3 = "complex.mul"(%0, %0) : (complex<f32>, complex<f32>) -> complex<f32>

// Tuple types (heterogeneous collections)
%4 = "test.create_tuple"() : () -> tuple<i32, f64, index>
%5 = "test.create_tuple"() : () -> tuple<tensor<2x2xf32>, memref<10xi32>, i1>

// Nested tuples
%6 = "test.create_tuple"() : () -> tuple<tuple<i32, i32>, f64>

// Empty tuple
%7 = "test.create_tuple"() : () -> tuple<>
