file: examples/demo-mini.scry
expr: { let l = List<Int>()
        var i = 0
        while i < 150 { l.push(i) i = i + 1 }
        l }
contains: "length":150,"truncated":true
