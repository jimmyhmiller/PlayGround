file: examples/demo-mini.scry
expr: class ScriptedModel { reply: String
  calls: Int = 42
  fn complete(prompt: String) -> String { self.reply } }
expr: ScriptedModel.instance(2)
contains: "calls":{"type":"Int","value":42}
contains: "reply":{"type":"String","value":"Looks good to me."}
