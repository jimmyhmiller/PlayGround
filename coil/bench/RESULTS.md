# Coil vs C benchmarks

Host: `Darwin 25.5.0 arm64`, Apple clang version 21.0.0 (clang-2100.1.1.101).
Coil = `coil build` (LLVM -O3). All programs verified to print identical results.

## `fib`  (result `102334155`)

| Command | Mean [ms] | Min [ms] | Max [ms] | Relative |
|:---|---:|---:|---:|---:|
| `C -O3` | 277.2 ± 2.7 | 274.9 | 283.5 | 1.00 |
| `Coil -O3` | 278.7 ± 1.8 | 275.9 | 280.8 | 1.01 ± 0.01 |

## `tak`  (result `9`)

| Command | Mean [ms] | Min [ms] | Max [ms] | Relative |
|:---|---:|---:|---:|---:|
| `C -O3` | 4.3 ± 0.4 | 3.8 | 4.9 | 1.25 ± 0.12 |
| `Coil -O3` | 3.4 ± 0.1 | 3.3 | 3.5 | 1.00 |

## `loop`  (result `11754921320367621504`)

| Command | Mean [ms] | Min [ms] | Max [ms] | Relative |
|:---|---:|---:|---:|---:|
| `C -O3` | 271.6 ± 4.1 | 266.0 | 276.5 | 1.00 |
| `Coil -O3` | 275.6 ± 8.7 | 265.5 | 290.0 | 1.01 ± 0.04 |

## `float`  (result `19.691043591914`)

| Command | Mean [ms] | Min [ms] | Max [ms] | Relative |
|:---|---:|---:|---:|---:|
| `C -O3` | 173.5 ± 3.7 | 169.7 | 180.6 | 1.00 |
| `Coil -O3` | 174.3 ± 3.6 | 169.4 | 181.5 | 1.00 ± 0.03 |

## `memory`  (result `8277958355204043136`)

| Command | Mean [ms] | Min [ms] | Max [ms] | Relative |
|:---|---:|---:|---:|---:|
| `C -O3` | 21.6 ± 0.4 | 21.0 | 22.3 | 1.00 |
| `Coil -O3` | 23.0 ± 0.4 | 22.6 | 23.6 | 1.07 ± 0.03 |

