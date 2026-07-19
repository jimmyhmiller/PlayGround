file: examples/redef.scry
expr: class Box { value: String
  tag: String = "T"
  fn show() -> String { self.tag + ":" + self.value } }
expr: Box.instance(0).show()
expr: Box.instance(0)
contains: "value":"T:hello"
contains: "tag":{"type":"String","value":"T"}
contains: "value":{"type":"String","value":"hello"}
