file: tests/run/modules_coexist.scry
expr: fields("Shell")
contains: "kind":"NotImplemented"
contains: ambiguous type name 'Shell'
contains: modpkg_a.shell.Shell
contains: modpkg_b.shell.Shell
