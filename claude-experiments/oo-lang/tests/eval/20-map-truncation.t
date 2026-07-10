file: examples/demo-mini.scry
expr: { let m = Map<Int,Int>()
        var i = 0
        while i < 130 { m.set(i, i) i = i + 1 }
        m }
contains: "length":130,"truncated":true
