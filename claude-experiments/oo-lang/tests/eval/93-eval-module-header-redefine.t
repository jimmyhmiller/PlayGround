file: tests/run/modules_coexist.scry
expr: module modpkg_b.shell
class Shell {
  label: String
  fn describe() -> String { "B2:" + self.label }
}
expr: module modpkg_b.shell
Shell(label: "z").describe()
expr: module modpkg_a.shell
Shell(label: "z").describe()
contains: {"value":{"type":"defined","defined":"Shell"
contains: {"value":{"type":"String","value":"B2:z"}}
contains: {"value":{"type":"String","value":"A:z"}}
