file: examples/demo-mini.scry
expr: fn makeAgent(nm: String, model: ScriptedModel, t1: Tool, t2: Tool) -> Agent {
  let inv = Inventory<Tool>(items: List<Tool>())
  let conv = Conversation(messages: Inventory<Message>(items: List<Message>()))
  Agent(name: nm, model: model, conversation: conv, tools: inv)
}
contains: "type":"defined"
contains: "defined":"makeAgent"
contains: "gen":1
