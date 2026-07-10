file: examples/redef.scry
expr: Counter.instance(0).get()
expr: Counter.instance(0).label()
expr: class Counter { n: Int
  fn get() -> Int { self.n + 100 }
  fn label() -> String { "COUNT!" } }
expr: Counter.instance(0).get()
expr: Counter.instance(0).label()
contains: {"type":"Int","value":5}
contains: "value":"count"
contains: "type":"defined"
contains: {"type":"Int","value":105}
contains: "value":"COUNT!"
