// Example: Control flow with successors (branches)
// Grammar: successor-list ::= `[` successor (`,` successor)* `]`
// Grammar: successor ::= caret-id (`:` block-arg-list)?

// Generic format with successors
%0 = "arith.constant"() <{value = 1 : i32}> : () -> i32
%1 = "arith.constant"() <{value = 0 : i32}> : () -> i32
%2 = "arith.cmpi"(%0, %1) <{predicate = 0 : i64}> : (i32, i32) -> i1
"cf.cond_br"(%2)[^bb1, ^bb2] : (i1) -> ()

^bb1:
  %3 = "arith.constant"() <{value = 42 : i32}> : () -> i32
  "cf.br"()[^bb3(%3 : i32)] : () -> ()

^bb2:
  %4 = "arith.constant"() <{value = 13 : i32}> : () -> i32
  "cf.br"()[^bb3(%4 : i32)] : () -> ()

^bb3(%arg0: i32):
  "func.return"(%arg0) : (i32) -> ()
