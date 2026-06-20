# Coil vs C benchmarks

Host: `Darwin 25.5.0 arm64`, Apple clang version 21.0.0 (clang-2100.1.1.101).
Coil = `coil build` (LLVM -O3). All programs verified to print identical results.

## `fib`  (result `102334155`)

| Command | Mean [ms] | Min [ms] | Max [ms] | Relative |
|:---|---:|---:|---:|---:|
| `C -O3` | 294.3 ± 2.7 | 289.8 | 298.3 | 1.00 |
| `Coil -O3` | 298.2 ± 2.6 | 294.9 | 302.2 | 1.01 ± 0.01 |

## `tak`  (result `9`)

| Command | Mean [ms] | Min [ms] | Max [ms] | Relative |
|:---|---:|---:|---:|---:|
| `C -O3` | 4.3 ± 0.5 | 3.9 | 5.3 | 1.00 |
| `Coil -O3` | 4.4 ± 0.4 | 3.9 | 4.9 | 1.02 ± 0.15 |

## `loop`  (result `11754921320367621504`)

| Command | Mean [ms] | Min [ms] | Max [ms] | Relative |
|:---|---:|---:|---:|---:|
| `C -O3` | 299.3 ± 3.4 | 294.0 | 304.3 | 1.00 |
| `Coil -O3` | 300.1 ± 2.7 | 296.4 | 304.2 | 1.00 ± 0.01 |

## `float`  (result `19.691043591914`)

| Command | Mean [ms] | Min [ms] | Max [ms] | Relative |
|:---|---:|---:|---:|---:|
| `C -O3` | 194.3 ± 2.3 | 191.4 | 197.9 | 1.01 ± 0.02 |
| `Coil -O3` | 191.7 ± 3.0 | 187.9 | 196.1 | 1.00 |

## `memory`  (result `8277958355204043136`)

| Command | Mean [ms] | Min [ms] | Max [ms] | Relative |
|:---|---:|---:|---:|---:|
| `C -O3` | 24.5 ± 1.0 | 23.5 | 26.1 | 1.05 ± 0.08 |
| `Coil -O3` | 23.4 ± 1.5 | 22.4 | 26.6 | 1.00 |

