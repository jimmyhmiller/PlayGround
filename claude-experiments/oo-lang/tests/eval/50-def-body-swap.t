file: examples/demo-mini.scry
expr: ScriptedModel.instance(0).complete("x")
expr: class ScriptedModel { reply: String
  fn complete(prompt: String) -> String { "TL;DR " + self.reply } }
expr: ScriptedModel.instance(0).complete("x")
contains: "value":"I will investigate that."
contains: "type":"defined"
contains: "value":"TL;DR I will investigate that."
