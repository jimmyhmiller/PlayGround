(module
  (import "main" "memory" (memory i64 1))
  (import "main" "__t" (table i64 4 funcref))
  (import "main" "host_double" (func $hd (param i64) (result i64)))
  (type $unary (func (param i64) (result i64)))
  (func (export "run") (param $argptr i64) (result i64)
    ;; read i64 at [argptr] (an arg the compiler placed in shared memory)
    (local $v i64) (local $r i64)
    local.get $argptr
    i64.load
    local.set $v
    ;; call back into the compiler two ways: direct import AND indirect via shared table slot 1
    local.get $v
    call $hd                       ;; direct imported call -> v*2
    i64.const 1
    call_indirect (type $unary)    ;; indirect via shared table slot 1 -> (v*2)*2
    local.set $r
    ;; write result back into shared memory at [argptr+8], prove shared write
    local.get $argptr
    i64.const 8
    i64.add
    local.get $r
    i64.store
    local.get $r))
