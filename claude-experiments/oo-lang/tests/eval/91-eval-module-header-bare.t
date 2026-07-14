file: tests/run/modules_coexist.scry
expr: Shell(label: "x").describe()
expr: module modpkg_b.shell
Shell(label: "x").describe()
contains: {"value":{"type":"String","value":"A:x"}}
contains: {"value":{"type":"String","value":"B:x"}}
