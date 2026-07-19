file: examples/demo-mini.scry
expr: class ScriptedModel { reply: String
  fn complete(prompt: String) -> String { self.reply + 5 } }
expr: ScriptedModel.instance(0).complete("x")
contains: "kind":"TypeError"
contains: expected String, found Int
contains: "value":"I will investigate that."
