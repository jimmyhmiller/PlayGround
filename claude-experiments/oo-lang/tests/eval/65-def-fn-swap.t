file: examples/redef.scry
expr: describe(Box.instance(0))
expr: fn describe(b: Box) -> String { "DESC: " + b.show() }
expr: describe(Box.instance(0))
contains: "value":"hello"
contains: "defined":"describe"
contains: "value":"DESC: hello"
