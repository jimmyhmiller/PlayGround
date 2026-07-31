file: examples/demo-mini.scry
expr: Process.exec("printf should-not-run")
readonly: yes
contains: "kind":"ReadOnly"
notcontains: should-not-run
