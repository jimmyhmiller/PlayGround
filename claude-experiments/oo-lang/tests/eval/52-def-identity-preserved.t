file: examples/demo-mini.scry
expr: class ScriptedModel { reply: String
  fn complete(prompt: String) -> String { self.reply } }
expr: ScriptedModel.instance(1)
contains: "ref":"ScriptedModel#1"
contains: "generation":0
contains: "reply":{"type":"String","value":"Here is the code."}
