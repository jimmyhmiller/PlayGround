file: examples/demo-mini.scry
expr: { let m = Map<String,Int>()
        m.set("a", 1)
        m.set("b", 2)
        m }
contains: "type":"map","keyType":"String","valueType":"Int","length":2
contains: [{"type":"String","value":"a"},{"type":"Int","value":1}]
