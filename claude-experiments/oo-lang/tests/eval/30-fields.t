file: examples/demo-mini.scry
expr: fields("Agent")
contains: {"name":"name","type":"String"}
contains: {"name":"tools","type":"ref:Inventory<Tool>"}
