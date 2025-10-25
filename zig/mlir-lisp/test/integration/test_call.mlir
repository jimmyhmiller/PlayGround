module {
  func.func @test() -> i64 {
    %c42 = arith.constant 42 : i64
    func.return %c42 : i64
  }

  func.func @main() -> i64 {
    %result = func.call @test() : () -> i64
    func.return %result : i64
  }
}
