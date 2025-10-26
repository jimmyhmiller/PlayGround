// Example: Type and attribute aliases
// Grammar: type-alias-def ::= `!` alias-name `=` type
// Grammar: attribute-alias-def ::= `#` alias-name `=` attribute-value

// Type aliases
!my_int = i32
!my_float = f64
!my_tensor = tensor<4x4xf32>
!my_memref = memref<16x16xf64>

// Attribute aliases
#zero_i32 = 0 : i32
#one_i32 = 1 : i32
#pi = 3.14159 : f64
#my_array = [1, 2, 3, 4] : tensor<4xi32>

// Using type aliases in operations
%0 = "arith.constant"() <{value = #zero_i32}> : () -> !my_int
%1 = "arith.constant"() <{value = #one_i32}> : () -> !my_int

// Using both type and attribute aliases
%2 = "arith.constant"() <{value = #pi}> : () -> !my_float

// Type alias in complex types
%3 = "tensor.empty"() : () -> !my_tensor
%4 = "memref.alloc"() : () -> !my_memref

// Attribute alias with complex values
#my_struct = {field1 = #zero_i32, field2 = #pi}
%5 = "test.op"() {value = #my_struct} : () -> !my_int
