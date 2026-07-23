(module
  (memory (export "memory") i64 1)
  (table (export "__t") i64 4 funcref)
  (func $host_double (export "host_double") (param i64) (result i64)
    local.get 0
    i64.const 2
    i64.mul)
  ;; place host_double at table slot 1 so a side module can call it indirectly too
  (elem (i64.const 1) func $host_double))
