file: tests/run/modules_coexist.scry
expr: modules()
contains: {"type":"Modules","children":[
contains: {"name":"modules_coexist","path":"modules_coexist","typeCount":0,"fnCount":1,"liveCount":0,"children":[]}
contains: {"name":"modpkg_a","path":"modpkg_a","typeCount":1,"fnCount":0,"liveCount":1,"children":[{"name":"shell","path":"modpkg_a.shell","typeCount":1,"fnCount":0,"liveCount":1,"children":[]}]}
contains: {"name":"modpkg_b","path":"modpkg_b","typeCount":1,"fnCount":0,"liveCount":1,"children":[{"name":"shell","path":"modpkg_b.shell","typeCount":1,"fnCount":0,"liveCount":1,"children":[]}]}
