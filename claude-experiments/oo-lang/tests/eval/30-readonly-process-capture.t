file: examples/demo-mini.scry
expr: Process.capture("printf should-not-run")
readonly: yes
contains: "kind":"ReadOnly"
notcontains: should-not-run
