file: examples/functions.scry
expr: trace(fib(nope))
expr: 3 * 4 + 2
expr: trace(fact(4))
contains: "kind":"TypeError"
contains: {"value":{"type":"Int","value":14}}
contains: {"fn":"fact","calls":4}
