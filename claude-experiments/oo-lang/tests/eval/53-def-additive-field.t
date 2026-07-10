file: examples/demo-mini.scry
expr: class ScriptedModel { reply: String
  calls: Int = 7
  fn complete(prompt: String) -> String { "calls=${self.calls} ${self.reply}" } }
expr: ScriptedModel.instance(0).complete("x")
contains: "type":"defined"
contains: "value":"calls=7 I will investigate that."
