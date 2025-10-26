// Test file for LLVM dialect features that might be problematic
// Tests: basic blocks, block arguments, conditional branches, pointer types with attrs

module {
  // Simple function that tests pointer equality and branches
  llvm.func @test_ptr_eq(%arg0: !llvm.ptr, %arg1: !llvm.ptr) -> i1 {
    %0 = llvm.icmp "eq" %arg0, %arg1 : !llvm.ptr
    llvm.return %0 : i1
  }

  // Function with multiple blocks, block arguments, and conditional branches
  llvm.func @test_control_flow(%cond: i1, %val: i32) -> i32 {
    llvm.cond_br %cond, ^bb1(%val : i32), ^bb2
  ^bb1(%arg: i32):  // Block with argument
    %c10 = llvm.mlir.constant(10 : i32) : i32
    %sum = llvm.add %arg, %c10 : i32
    llvm.br ^bb3(%sum : i32)
  ^bb2:  // Block without arguments
    %c20 = llvm.mlir.constant(20 : i32) : i32
    llvm.br ^bb3(%c20 : i32)
  ^bb3(%result: i32):  // Common successor with block argument
    llvm.return %result : i32
  }

  // Function demonstrating pointer attributes and function calls
  llvm.func @test_ptr_attrs(%ptr: !llvm.ptr {llvm.readonly}) -> i32 {
    %0 = llvm.load %ptr : !llvm.ptr -> i32
    %1 = llvm.mlir.constant(5 : i32) : i32
    %2 = llvm.add %0, %1 : i32
    llvm.return %2 : i32
  }

  // Function that calls another function
  llvm.func @test_call(%x: i32) -> i32 {
    %ptr = llvm.mlir.zero : !llvm.ptr
    %val = llvm.mlir.constant(42 : i32) : i32
    %eq = llvm.call @test_ptr_eq(%ptr, %ptr) : (!llvm.ptr, !llvm.ptr) -> i1
    llvm.cond_br %eq, ^bb_then, ^bb_else
  ^bb_then:
    llvm.return %val : i32
  ^bb_else:
    llvm.return %x : i32
  }
}
