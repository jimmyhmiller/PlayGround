file: tests/run/modules_coexist.scry
expr: fields("modpkg_a.shell.Shell")
expr: methods("modpkg_b.shell.Shell")
contains: {"value":[{"name":"label","type":"String"}]}
contains: {"value":[{"name":"describe","params":[],"returns":"String"}]}
