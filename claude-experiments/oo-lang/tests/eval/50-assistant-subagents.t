file: examples/assistant.scry
stdin: research quantum\n
expr: Agent.instances()
expr: Message.instances()
expr: ScriptedModel.instances()
contains: "elementType":"Agent","length":3
contains: "name":{"type":"String","value":"researcher"}
contains: "name":{"type":"String","value":"summarizer"}
contains: "status":{"type":"AgentStatus","case":"Done"}
contains: "elementType":"Message","length":21
contains: "elementType":"ScriptedModel","length":3
