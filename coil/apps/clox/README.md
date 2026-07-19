# clox in Coil

A port of **clox** — the bytecode virtual machine from Part III of Bob Nystrom's
*[Crafting Interpreters](https://craftinginterpreters.com/)* — to the Coil language.
A single ~1400-line Coil file implementing a complete Lox interpreter: scanner,
single-pass Pratt compiler, bytecode VM, closures/upvalues, classes, inheritance,
`super`, string interning, open-addressing hash tables, and a tri-color
**mark-sweep garbage collector**.

Observable behavior is byte-identical to the reference C clox: all output goes
through libc `write(1)/write(2)` + `snprintf("%g", …)`, so numbers, error
messages, stack traces, and exit codes (65 compile / 70 runtime) match exactly.

## Build & run

Build with the **self-hosted** `./coil` (it has the character-literal syntax the
scanner uses — the Rust reference compiler does not):

```sh
# from the repo root
./coil build apps/clox/clox.coil -o /tmp/clox
/tmp/clox path/to/script.lox
```

## Test suite

`run-tests.py` faithfully replicates the Crafting Interpreters Dart test harness
(`// expect:` / `// expect runtime error:` / `// [line N] Error` comments, exit
codes). The `tests/lox/` directory vendors the **clox suite** — every test except
`scanning/` and `expressions/` (which need standalone modes clox doesn't have),
**246 tests**.

```sh
cd apps/clox
python3 run-tests.py /tmp/clox          # -> 246 passed, 0 failed
```

### Stress-testing the GC

Set `CLOX_GC_STRESS=1` to collect on **every** allocation (clox's
`DEBUG_STRESS_GC`) — the strongest test that the collector never frees a live
object:

```sh
CLOX_GC_STRESS=1 python3 run-tests.py /tmp/clox   # -> 246 passed, 0 failed
```

Verified: full suite green with GC on and under stress; memory stays bounded
(~10 MB for 2M allocations), confirming reclamation.
