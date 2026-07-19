file: examples/demo-mini.scry
expr: class ScriptedModel {
  fn complete(prompt: String) -> String { "hi" } }
expr: ScriptedModel.instance(0).complete("x")
contains: "kind":"NotImplemented"
contains: field 'reply' was removed
contains: 3 live instance
contains: "value":"I will investigate that."
