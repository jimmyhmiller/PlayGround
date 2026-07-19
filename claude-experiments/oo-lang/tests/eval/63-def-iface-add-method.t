file: examples/demo-mini.scry
expr: interface Tool {
  fn call(input: String) -> String
  fn cancel() -> Void
}
contains: "kind":"TypeError"
contains: does not implement method 'cancel'
