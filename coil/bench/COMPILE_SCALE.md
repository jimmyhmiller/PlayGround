# Compile speed at scale

Host: `Darwin 25.5.0 arm64`. Generated programs: a reachable chain of `gN` functions
(~1 fn/line), 10% generic (instantiated at i64+f64), 20% using the `when` macro.
Best of 3 runs. **front-end** = `coil emit-ir` (read→expand→check→mono→IR, no
optimizer); **build** = full `coil build` (adds the LLVM -O3 pipeline + link).

| functions | LOC | front-end (s) | build (s) | front-end ms/fn | build ms/fn |
|---:|---:|---:|---:|---:|---:|
| 250 | 276 | 0.053 | 0.146 | 0.212 | 0.584 |
| 1000 | 1101 | 0.070 | 0.346 | 0.070 | 0.346 |
| 4000 | 4401 | 0.145 | 1.114 | 0.036 | 0.279 |
| 8000 | 8801 | 0.250 | 1.771 | 0.031 | 0.221 |
