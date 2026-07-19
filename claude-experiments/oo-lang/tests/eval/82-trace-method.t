file: examples/demo-mini.scry
expr: trace(Agent.at(0,0).step("hi"))
contains: "type":"Trace"
contains: {"fn":"step","calls":1}
contains: {"fn":"append","calls":2}
contains: "fn":"step","args":[{"type":"ref","class":"Agent"
