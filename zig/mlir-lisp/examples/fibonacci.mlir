// Recursive Fibonacci Function in MLIR
//
// Function signature: fibonacci(n: i32) -> i32
//
// Algorithm:
//   if n <= 1:
//     return n
//   else:
//     return fibonacci(n-1) + fibonacci(n-2)

module {
  func.func @fibonacci(%n: i32) -> i32 {
    // Check if n <= 1 (base case)
    %c1 = arith.constant 1 : i32
    %cond = arith.cmpi sle, %n, %c1 : i32

    // scf.if with then/else branches
    %result = scf.if %cond -> (i32) {
      // Then region: base case, return n
      scf.yield %n : i32
    } else {
      // Else region: recursive case, return fib(n-1) + fib(n-2)

      // Compute fib(n-1)
      %c1_rec = arith.constant 1 : i32
      %n_minus_1 = arith.subi %n, %c1_rec : i32
      %fib_n_minus_1 = func.call @fibonacci(%n_minus_1) : (i32) -> i32

      // Compute fib(n-2)
      %c2 = arith.constant 2 : i32
      %n_minus_2 = arith.subi %n, %c2 : i32
      %fib_n_minus_2 = func.call @fibonacci(%n_minus_2) : (i32) -> i32

      // Add fib(n-1) + fib(n-2) and yield
      %sum = arith.addi %fib_n_minus_1, %fib_n_minus_2 : i32
      scf.yield %sum : i32
    }

    // Return the result
    func.return %result : i32
  }
}
