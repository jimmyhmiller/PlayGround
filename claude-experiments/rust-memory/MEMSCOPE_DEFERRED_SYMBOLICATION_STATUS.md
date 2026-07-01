# memscope: deferred symbolication — status & open issues

> Working notes from making `record_to_file` cheap enough to profile a real
> workload (Turbopack building an app). **Do not commit** — scratch status doc.

## TL;DR

- **Fixed (the headline goal):** the traced process no longer does *any* DWARF
  work. It records raw frame IPs + a load slide; symbolication (stacks + type
  recovery) happens at **read time** in `replay`/`analyze`/`diff`. For a 309-module
  Turbopack build this took the traced process from **2.12 GB → 279 MB**, **~38s →
  4s**, and the recording file from **8.6 GB → 146 MB**, with **identical** recovered
  types.
- **Partially fixed:** read-time `analyze` **time** went **84s → ~9s** (per-unique-IP
  dedup + two-pass).
- **Open issue:** read-time `analyze` **memory** is still **~12.4 GB**. Fully
  diagnosed (below); the correct fix (a targeted gimli inline walk) was attempted
  twice and not landed, so the code is left on the correct-but-heavy addr2line path.

---

## Background — why we touched this

Goal was to memory-analyze an example app with the `memscope` allocator tooling.
Target: `turbopack-cli build` on a generated test app, with `memscope::MemScope`
swapped in as the global allocator (in a worktree) and `MEMSCOPE_RECORD=<path>`
recording the full allocation stream.

First run was alarming: **2.12 GB peak / very slow / 8.6 GB recording** for a build
that takes **~130 MB / 0.4s** uninstrumented. The expectation was that recording
streams to disk and stays lean — it did not.

## Root cause of the traced-process overhead

`FileRecorder::create` built `TypeOracle::for_current_process()` **inside the
traced process**:

- `dsymutil` produces an **843 MB** `.dSYM`; the whole thing is `std::fs::read`
  into a `Vec<u8>`.
- `dwarf::build` walks all of it into `HashMap<String, FnTypeInfo>` (+ layout maps)
  — owned strings for every monomorphized function.
- Each newly-seen site was symbolicated **on the pump thread** via the in-process
  `backtrace` crate.

Confirmed by gating the oracle off (`MEMSCOPE_RAW`): **287 MB / 3.25s / 98 MB file**.
So in-process symbolication accounted for **~1.85 GB of memory, ~57s of CPU, and
8.5 GB of file bloat** (resolved site records embed deep frames × long names).

## The fix — defer symbolication to read time

The traced process writes **raw IPs**; the reader resolves them against the dSYM
once, off the hot path.

Mechanism detail that drove the design: `resolve_site_ips` does two things —
1. frames/file/line via the `backtrace` crate (works **only in the live process**,
   ASLR-aware);
2. type recovery (`Boxed<Particle>`) via the gimli index, keyed by **mangled name**.

Read-time can't use the `backtrace` crate (no live process), so the reader
symbolizes **by address** with `addr2line` against the on-disk dSYM. To map a
recorded runtime return address to a static (link-time) address we record the
main image's **ASLR load slide** in the header (`static = ip - slide`).

### Changes (rust-memory)

| Crate | Change |
|---|---|
| `memscope-proto` (`recfmt.rs`) | header `version` → 2 + `slide: u64`; new `TAG_RSITE` raw-site record (`id` + `u16 n` + `n × u64` ips); encode/decode + tests |
| `memscope-symbols` (`load.rs`) | `current_image_slide()` (macOS `_dyld_get_image_vmaddr_slide(0)`) |
| `memscope-symbols` (`lib.rs`) | `resolve_raw_sites(exe, slide, sites)` — address-based addr2line resolution + reuse of the recognizer; `finish_frames` shared with the in-process path |
| `memscope-agent` (`record.rs`) | `FileRecorder` drops the oracle; writes `TAG_RSITE` (binary) / `{"rsite","ips"}` (JSON) + slide in header |
| `memscope-replay` (`lib.rs`) | parse slide + `TAG_RSITE`; resolve once at read time (binary + JSON readers) |
| `memscope-cli` (`main.rs`) | the standalone `replay_binary`/`replay_json` decoders also parse slide + raw sites + resolve; mimalloc global allocator |

### Result (309-module Turbopack build)

| | before | after |
|---|---|---|
| traced-process peak | 2.12 GB | **279 MB** |
| wall (default threads) | stalled/slow | **4.0 s** |
| recording file | 8.6 GB | 146 MB |
| recovered types | yes | **yes — identical** (`Boxed<(TaskId, …TaskStorage)>`, etc.) |

The remaining ~150 MB over the 130 MB uninstrumented baseline is genuine
tracking-buffer overhead (rings + live set) — the "stream to disk, stay lean"
behavior we wanted.

---

## Open issue — read-time `analyze` memory (~12.4 GB)

Resolving the 146 MB recording with `memscope analyze` peaks at **~12.4 GB**. Time
was fixed (84s → ~9s) but memory was not.

### What it is (precisely diagnosed)

Peak **scales with the number of distinct addresses resolved**:

| addresses resolved | peak footprint |
|---|---|
| 0 (type index only) | 2.1 GB |
| 100 | 2.1 GB |
| 2,000 | 3.3 GB |
| 10,081 (all) | 12.6 GB |

≈ **0.6 MB per unique IP**. The recording has **116,380 sites / 6.3 M frame-refs**
but only **10,081 unique IPs** (avg stack depth 54). The cost is addr2line's
`find_frames`: it parses and **retains the entire inline-frame tree of every
function it touches**, and Turbopack inlines extremely deeply, so a single
function can balloon to gigabytes transiently — even though we keep only a few
frames (the assembled `ip_frames` cache is just **26 MB**).

### What it is NOT (ruled out)

- **Not the type index** — `dwarf::build` alone is ~2.1 GB (matches the in-process
  oracle), measured by skipping resolution.
- **Not the result** — `ip_frames` totals 26 MB (10k IPs, 57k frames).
- **Not allocator retention** — adding mimalloc + chunking the context (drop every
  N lookups) did **not** lower peak; smaller chunks were *slower* with the *same*
  peak. It's a genuine transient parse, not freed-but-unreturned memory.

### Things tried

| approach | result |
|---|---|
| per-unique-IP dedup + two-pass (drop ctx before type index) | **time 84s → ~9s**, memory unchanged → **shipped** |
| chunk the addr2line context (drop every 1–2k lookups) | no memory change; small chunks slower (rebuilds the index) |
| mimalloc + eager purge (`MIMALLOC_PURGE_DELAY=0`) | no change (memory is live during parse, not retained) |
| symbol-table names + `find_location` (no `find_frames`) | **5 GB**, but **breaks type recovery** — recognizer needs the inlined `Box`/`Vec` frames → **0 findings**. Unacceptable. |
| custom gimli walk, recursive active-set prune | **72 GB** (quadratic filtering + per-DIE Vec allocs across all units) — abandoned |
| custom gimli walk, per-unit subprogram table → per-function subtree | **11 GB + broke types** — the per-unit full DFS to collect subprograms was itself heavy and the inline-path collection produced no container frames — abandoned |

### Recommended fix (next session)

A **targeted DWARF inline walk** that, for each probe, descends only the inline
subroutines whose PC range covers it (one root→leaf path), never materializing
whole-function inline trees. The idea is right; both attempts failed on execution.
Do it test-first:

1. Small gimli unit test on a tiny binary with known inlining; assert the frame
   chain matches `addr2line` for a handful of addresses.
2. Cheap per-unit subprogram range table (scalar `low/high_pc`, no inline parse),
   binary-search probe → function offset.
3. `entries_tree(Some(func_off))`, descend following only covering
   `inlined_subroutine`/`lexical_block` ranges; resolve names via
   `DW_AT_abstract_origin`; file/line via a filtered line-program pass keeping
   only target rows.
4. Verify memory (~2–3 GB target) **and** that `analyze` findings match the
   addr2line output before replacing it.

Alternative / cheaper levers if the walk stays hard:
- Build `turbopack-cli` with lighter debuginfo (`debug = 1` / line-tables) — smaller
  dSYM, cheaper addr2line, at some cost to template-parameter type recovery.
- Fold function-range + name collection into the existing single `dwarf::build`
  DIE walk (it already visits every DIE) instead of a second pass.

---

## Current code state

- **Working / correct.** Whole workspace builds; `memscope-proto`,
  `memscope-replay`, `memscope-symbols` tests pass (24).
- Read-time path is on **addr2line** (`symbolicate_addr` in `lib.rs`): correct
  types, ~9s, **~12.4 GB**.
- No custom resolver, lean mode, chunking, or debug env-knobs remain (all reverted);
  mimalloc global allocator kept in the CLI (harmless, mildly helpful).
- Turbopack-side instrumentation lives only on the **`memscope-analysis` git
  worktree** of next.js (global-allocator swap + `MEMSCOPE_RECORD` hook in
  `turbopack-cli/src/main.rs`); not on any real branch.

### How to reproduce

```sh
# in the next.js memscope-analysis worktree
cargo build --profile release-with-debug --no-default-features -p turbopack-cli -p turbopack-create-test-app
./target/release-with-debug/turbopack-create-test-app --modules 300 --directories 30 --package-json /tmp/app
(cd /tmp/app && npm i --no-audit --no-fund react@18 react-dom@18)
MEMSCOPE_RECORD=/tmp/tp.mscope ./target/release-with-debug/turbopack-cli build src/index.jsx --dir /tmp/app --target browser

# in rust-memory
cargo build --release
/usr/bin/time -l ./target/release/memscope analyze /tmp/tp.mscope --top 10   # ~9s, ~12.4 GB
./target/release/memscope replay /tmp/tp.mscope                              # type-resolved heap by type
```
