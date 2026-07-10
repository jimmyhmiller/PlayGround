file: examples/redef.scry
expr: class Counter { count: Int
  fn get() -> Int { self.count }
  fn label() -> String { "count" } }
contains: "kind":"NotImplemented"
contains: renamed
