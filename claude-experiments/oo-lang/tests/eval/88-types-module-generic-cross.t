file: tests/run/modgen_main.scry
expr: types()
contains: "name":"Item","module":"modgen_b.item","qualified":"modgen_b.item.Item"
contains: "name":"Box<Item>","module":"modgen_a.box","qualified":"modgen_a.box.Box<modgen_b.item.Item>"
