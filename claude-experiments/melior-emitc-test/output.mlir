module {
  emitc.func @main() -> i32 {
    %0 = "emitc.constant"() <{value = 10 : i32}> : () -> i32
    %1 = "emitc.constant"() <{value = 32 : i32}> : () -> i32
    %2 = "emitc.constant"() <{value = 42 : i32}> : () -> i32
    return %2 : i32
  }
}
