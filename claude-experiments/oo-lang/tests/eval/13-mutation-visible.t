file: examples/demo-mini.scry
expr: { let a = Agent.at(0,0)
        let before = a.conversation.size()
        a.step("hi")
        a.conversation.size() - before }
contains: {"value":{"type":"Int","value":2}}
