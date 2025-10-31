// Basic block with label
// Grammar: block ::= block-label operation+
func.func @test() {
  ^entry:
    %0 = arith.constant 42 : i32
    func.return
}
