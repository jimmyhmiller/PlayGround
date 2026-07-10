file: examples/demo-mini.scry
expr: class ScriptedModel { reply: String
  fn complete(prompt: String) -> String { self.reply }
  fn extra() -> Int { 1 } }
contains: "kind":"NotImplemented"
contains: cannot add method 'extra'
