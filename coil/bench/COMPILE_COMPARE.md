# Compiler throughput — Coil vs clang vs zig

Host: `Darwin 25.5.0 arm64`, Apple clang version 21.0.0 (clang-2100.1.1.101), zig 0.16.0.
Coil compiler built `--release`. Equivalent programs (a reachable chain of N plain
`g_i` integer functions, no generics/macros) built to a full executable, best of 3
runs, lower is faster. Coil has no distinct Debug/Release build *modes* yet, so its
"debug" column is `build -g` (the light, non-O3 pipeline + DWARF).

## Release builds

| functions | LOC | coil (s) | clang (s) | zig (s) | coil ms/fn | clang ms/fn | zig ms/fn |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 250 | 252 | 0.092 | 0.085 | 0.161 | 0.368 | 0.340 | 0.644 |
| 1000 | 1002 | 0.136 | 0.122 | 0.209 | 0.136 | 0.122 | 0.209 |
| 4000 | 4002 | 0.344 | 0.298 | 0.448 | 0.086 | 0.074 | 0.112 |

## Debug builds

| functions | LOC | coil (s) | clang (s) | zig (s) | coil ms/fn | clang ms/fn | zig ms/fn |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 250 | 252 | 0.123 | 0.104 | 1.214 | 0.492 | 0.416 | 4.856 |
| 1000 | 1002 | 0.188 | 0.155 | 1.264 | 0.188 | 0.155 | 1.264 |
| 4000 | 4002 | 0.397 | 0.326 | 1.462 | 0.099 | 0.082 | 0.365 |
