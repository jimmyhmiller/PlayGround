file: examples/demo-mini.scry
expr: class ScriptedModel { reply: String
  fn complete(prompt: String) -> String { "NEW: " + self.reply } }
expr: ScriptedModel.instance(0).complete("x")
contains: "type":"defined"
contains: "defined":"ScriptedModel"
contains: "gen":1
contains: NEW: I will investigate that.
