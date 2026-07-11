file: examples/bank.scry
expr: schema()
contains: "name":"Transaction"
contains: "ctor":[{"name":"kind","type":"String"},{"name":"amount","type":"Int"},{"name":"note","type":"String"}]
contains: "ctor":[{"name":"owner","type":"Customer"},{"name":"kind","type":"Kind"}]
