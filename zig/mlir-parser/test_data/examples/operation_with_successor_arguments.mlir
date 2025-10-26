"test.op"() ({
^bb0:
  %arg0 = "test.arg"() : () -> i32
  "cf.br"(%arg0) [^bb1] : (i32) -> ()
^bb1(%arg1: i32):
  "test.return"(%arg1) : (i32) -> ()
}) : () -> ()
