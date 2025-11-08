;; Minimal GPU test
(operation
  (name builtin.module)
  (attributes {:gpu.container_module true})
  (regions
    (region
      (block
        (operation
          (name gpu.module)
          (attributes {:sym_name @test})
          (regions (region (block))))))))
