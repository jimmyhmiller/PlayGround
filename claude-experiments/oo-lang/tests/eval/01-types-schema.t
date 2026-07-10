file: examples/demo-mini.scry
expr: types()
contains: "elementType":"TypeDescriptor"
contains: "name":"Agent"
contains: "liveCount":3
contains: {"name":"conversation","type":"ref:Conversation"}
contains: {"name":"tools","type":"ref:Inventory<Tool>"}
contains: "returns":"Result<Tool,String>"
