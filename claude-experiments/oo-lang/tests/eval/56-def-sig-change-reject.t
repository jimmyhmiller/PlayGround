file: examples/demo-mini.scry
expr: class ScriptedModel { reply: String
  fn complete(prompt: String, n: Int) -> String { self.reply } }
expr: ScriptedModel.instance(0).complete("x")
contains: "kind":"NotImplemented"
contains: cannot change the signature
contains: "value":"I will investigate that."
