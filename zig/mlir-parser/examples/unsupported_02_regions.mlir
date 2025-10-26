// Example: Operations with regions (scf.if, scf.for, etc.)
// Grammar: region-list ::= `(` region (`,` region)* `)`
// Grammar: region ::= `{` entry-block? block* `}`

// scf.if with two regions (then and else)
%cond = "arith.constant"() <{value = 1 : i1}> : () -> i1
%result = "scf.if"(%cond) ({
  %then_val = "arith.constant"() <{value = 42 : i32}> : () -> i32
  "scf.yield"(%then_val) : (i32) -> ()
}, {
  %else_val = "arith.constant"() <{value = 0 : i32}> : () -> i32
  "scf.yield"(%else_val) : (i32) -> ()
}) : (i1) -> i32

// scf.for with one region (loop body)
%lb = "arith.constant"() <{value = 0 : i32}> : () -> i32
%ub = "arith.constant"() <{value = 10 : i32}> : () -> i32
%step = "arith.constant"() <{value = 1 : i32}> : () -> i32
%init = "arith.constant"() <{value = 0 : i32}> : () -> i32

%loop_result = "scf.for"(%lb, %ub, %step, %init) ({
^bb0(%iv: i32, %iter_arg: i32):
  %sum = "arith.addi"(%iter_arg, %iv) : (i32, i32) -> i32
  "scf.yield"(%sum) : (i32) -> ()
}) : (i32, i32, i32, i32) -> i32
