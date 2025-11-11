(mlsp.module {
  (mlsp.func @fibonacci {
    (attributes {
      :sym_name @fibonacci
      :function_type (!function (inputs !i32) (results !i32))
    })
    (mlsp.region {
      ^bb0(%arg0: !i32):
        (mlsp.value %0 {
          :type !i1
        } (mlsp.cmp {
          :predicate eq
        } %arg0 (%mlsp.constant 0)))
    })
  })
})
