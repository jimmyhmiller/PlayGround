// Example: Pretty dialect types and attributes
// Grammar: pretty-dialect-type ::= dialect-namespace `.` pretty-dialect-type-lead-ident dialect-type-body?
// Grammar: pretty-dialect-attribute ::= dialect-namespace `.` pretty-dialect-attribute-lead-ident dialect-attribute-body?

// LLVM dialect types (pretty format)
%0 = "test.op"() : () -> !llvm.ptr
%1 = "test.op"() : () -> !llvm.ptr<i32>
%2 = "test.op"() : () -> !llvm.array<10 x i32>
%3 = "test.op"() : () -> !llvm.struct<(i32, f64, ptr)>

// LLVM dialect attributes (pretty format)
%4 = "test.op"() {linkage = #llvm.linkage<internal>} : () -> i32
%5 = "test.op"() {cc = #llvm.cconv<fastcc>} : () -> i32

// Custom dialect types
%6 = "test.op"() : () -> !mydialect.custom_type<42>
%7 = "test.op"() : () -> !mydialect.parametric_type<i32, f64>

// Custom dialect attributes
%8 = "test.op"() {attr = #mydialect.my_attr<"value">} : () -> i32
%9 = "test.op"() {attr = #mydialect.complex_attr<{a = 1, b = 2.0}>} : () -> i32

// Nested dialect types and attributes
%10 = "test.op"() : () -> !mydialect.container<!llvm.ptr<i32>>
%11 = "test.op"() {attr = #mydialect.wrapper<#llvm.linkage<external>>} : () -> i32
