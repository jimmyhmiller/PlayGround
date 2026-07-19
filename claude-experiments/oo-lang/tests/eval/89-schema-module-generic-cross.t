file: tests/run/modgen_main.scry
expr: schema()
contains: "name":"Box<Item>","module":"modgen_a.box","qualified":"modgen_a.box.Box<modgen_b.item.Item>"
contains: {"name":"items","type":"list:Item","refTypes":["Item"],"refQualified":["modgen_b.item.Item"]}
contains: "name":"HttpResponse","module":"","qualified":"HttpResponse"
