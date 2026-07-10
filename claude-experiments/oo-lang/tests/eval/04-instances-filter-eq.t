file: examples/demo-mini.scry
expr: Agent.instances(filter: "name == \"coder\"")
contains: "length":1
contains: "value":"coder"
notcontains: "researcher"
