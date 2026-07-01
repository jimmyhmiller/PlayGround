# Constant-memory symbolication — plan & test strategy

> Goal: resolve N recorded return addresses to typed, inlined, source-located
> frames using memory **bounded by the binary's debug info, not by N** — and not
> blowing up on deeply-inlined functions. Replaces the addr2line path that hit
> **68 GB** on a real (1.8 GB) recording. **Do not commit** — working plan.

## 0. The invariant we are building to

> **Peak resident memory while resolving `N` addresses is `O(binary) + O(max
> inline depth) + O(output)` and independent of `N`.**

Concretely: `peak_rss(N=100_000) − peak_rss(N=100)` is a small constant (target
< 256 MB), not gigabytes. This invariant is a **test** (§4.D), not a hope.

## 1. Why the current path is not constant (recap)

Measured: synthetic 146 MB recording → 12.6 GB; real 1.8 GB recording → 68 GB (OOM).
Three independent causes, all removable:

1. **addr2line retains whole inline trees.** `Context::find_frames` parses and
   caches, for the Context's lifetime, the *entire* inlined-subroutine tree of
   every function it touches. We only need the **one root→leaf path covering the
   probe**. Cost scales with (touched functions × their full inline-tree size).
2. **dSYM read into a `Vec<u8>`** (~0.8–1 GB resident). Should be `mmap`.
3. **Global type index** (`HashMap<linkage_name, FnTypeInfo>`) holds owned
   strings for *every* function (~2 GB). Template params can come from the
   covering function's own DIE instead.

## 2. Design

A new `addr_resolve` module in `memscope-symbols` (replaces the deleted attempt).

### 2.1 Inputs / outputs (unchanged contract)
`resolve(exe, slide, ips: &[u64]) -> HashMap<u64, Vec<SymFrame>>`, where each
`SymFrame` is `{ ip, mangled, demangled, file, line, inlined }`, innermost-first,
real function last — **byte-for-byte the same shape the addr2line path produces**
(so the recognizer/`finish_frames` are unchanged downstream).

### 2.2 Components
1. **mmap the dSYM** (`memmap2`), parse with `object` borrowing the mmap. No
   `Vec<u8>`. The OS pages debug sections in/out; resident stays low.
2. **Function table (built once):** stream every unit's top-level
   `DW_TAG_subprogram`, recording `(low_pc, high_pc, unit_idx, die_offset)` into a
   `Vec`, sorted by `low_pc`. Memory = O(#functions) × ~32 B (a few MB for a huge
   binary). Discontiguous functions: store each range. This is the only
   per-binary structure; it does **not** grow with N.
   - Alternative considered: `.debug_aranges` for addr→unit. Rust dSYMs don't
     reliably emit it, so we build our own table. (Decision: own table.)
3. **Per-probe resolution (constant transient):**
   - binary-search the table → `(unit, subprogram DIE)`.
   - parse that **one unit** lazily (gimli `Unit`); reuse across probes in the
     same unit by caching **only the current unit** (evict when the next probe is
     in a different unit — addresses are processed **sorted**, so we touch each
     unit once and hold at most one parsed unit at a time).
   - descend the subprogram subtree following **only** children whose pc-range
     covers the probe (a single inline path). Emit a frame per
     `subprogram`/`inlined_subroutine` on that path. Never visit sibling inline
     subtrees — this is the whole-tree parse we are eliminating.
   - names: `DW_AT_linkage_name`/`DW_AT_name`, inlined via `DW_AT_abstract_origin`.
   - inline frame `file:line`: `DW_AT_call_file`/`DW_AT_call_line`; innermost
     (real) frame's line from the line program (§2.4).
   - retain nothing after pushing the frames for this probe.
4. **Type recovery without the global index:** read the covering subprogram's own
   `DW_TAG_template_type_parameter` children (`T → u64`, etc.) during the same
   walk, feed the recognizer. No `HashMap<name, …>` over all functions.
5. **Line program (§2.4):** per unit, build a compact sorted `Vec<(addr, file,
   line)>` lazily; resolve all this unit's probes against it; drop it when the
   unit changes. One unit's line rows at a time — bounded.

### 2.3 Processing order
Sort probes by static address up front. Then a single linear sweep:
- group consecutive probes by unit → parse unit once, build its line index once,
  resolve all its probes, drop unit + line index. **At most one unit materialized
  at any instant.** This is what makes it constant regardless of N.

### 2.4 macOS dSYM specifics
- `static = ip - slide` (slide already recorded per-image, dylib-aware — done).
- dSYM is a separate Mach-O; `object` parses it; DWARF addresses are link-time.
- Probe call site: use `static - 1` (land inside the call), as today.

## 3. Implementation phases (each independently testable, lands behind the diff oracle)

- **Phase 0 — clean slate + oracle.** Revert the lazy-subset edits to
  `memscope-replay` (restore `read_recording` resolving all; the new symbolizer
  makes that cheap again). Keep the addr2line resolver as
  `resolve_raw_sites_addr2line` (test-only oracle). Add the differential harness
  (§4.A).
- **Phase 1 — mmap.** Swap `fs::read(dsym)` → `memmap2::Mmap`. Existing tests
  pass; baseline RSS drops ~dSYM size. (Measurable, isolated.)
- **Phase 2 — function table.** Build + binary-search. Test: every probe maps to
  the function addr2line attributes it to (§4.A, function-only).
- **Phase 3 — inline-path walk** (names + inline flags + call_file/line). Tests
  §4.A (frames) + §4.B (hermetic fixture).
- **Phase 4 — line program** (innermost file:line). §4.A includes file:line.
- **Phase 5 — template-param type recovery** (drop the global index for this
  path). §4.A (shape+type) + §4.C (analyze findings match).
- **Phase 6 — make it the default** `resolve_raw_sites`; addr2line stays compiled
  for tests only. Run §4.D (memory invariant).
- **Phase 7 — real recording.** Run on the 1.8 GB build recording; assert
  completes under a hard memory cap and yields sane findings (§4.E).

## 4. Test strategy (this is the point)

### 4.A Differential correctness vs addr2line (the oracle)
addr2line is the known-correct reference. Test: for a sample of real addresses,
`resolve()` must equal `resolve_raw_sites_addr2line()`.
- **Fixtures:** addresses pulled from a *small* committed recording produced by
  the repo's `demo` crate (hermetic, ~MBs), plus an *ignored* test that runs
  against the turbopack recordings in `/tmp` when present.
- **Assertions, per site:**
  - recovered `(shape, element_type)` **equal** (headline — must match);
  - boundary (innermost non-runtime) **function name equal**;
  - boundary **file:line equal**;
  - inline-frame **count equal**, and each frame's demangled name equal.
- **Acceptance:** 100% match on `(shape, element_type)` and boundary name/loc;
  ≥99% on full inline-frame equality (a few addr2line tie-breaks on
  zero-length ranges may differ — each mismatch is printed and reviewed, not
  silently tolerated). Mismatches fail the test with a diff.

### 4.B Hermetic inline-structure unit test
Commit a tiny crate `tests/fixtures/inline_fixture` with **controlled** inlining:
```rust
#[inline(always)] fn a() -> Box<u64> { b() }
#[inline(always)] fn b() -> Box<u64> { Box::new(7) }   // alloc site
#[inline(never)]  fn entry() -> Box<u64> { a() }
```
Build with `debug=2`, find the alloc instruction address (via the symbol +
known offset, or a recorded run), and assert the resolved chain is
`Box::new` (inlined) → `b` (inlined) → `a` (inlined) → `entry` (real), with the
right `file:line` per frame and `element_type = u64`, `shape = Boxed`. This
proves the path walk + template recovery on a case we fully control.

### 4.C End-to-end findings parity
`analyze` on the committed small recording (and, ignored, the 146 MB synthetic)
must produce the **same findings** (detector, type, location, evidence counts)
with the new symbolizer as with the addr2line oracle. Snapshot the top-N findings
JSON from the oracle once; assert the new path reproduces it.

### 4.D Memory invariant (the core test)
A `examples/resolve_mem.rs`: `resolve_mem <recording> <N>` resolves the first `N`
distinct addresses and prints peak RSS (`getrusage(RUSAGE_SELF).ru_maxrss`).
A test (or CI script) runs it as a **subprocess** for `N ∈ {100, 1k, 10k, 100k}`
(capped at the recording's distinct count) and asserts:
```
peak_rss(N_max) - peak_rss(100)  <  MEM_SLACK   (target 256 MB)
```
Run as subprocesses so each peak is clean (ru_maxrss is monotonic within a
process). This is the test that *defines* "constant memory." It runs against the
turbopack recordings when present (ignored otherwise), and against a synthesized
many-address recording so it's reproducible in CI.

### 4.E Real-recording acceptance
Ignored integration test / script: `analyze` the 1.8 GB build recording under a
hard cap (e.g. `ulimit`-style or measured peak) — **must stay < 4 GB** and finish,
versus the 68 GB OOM today. Output sanity: nonzero findings, plausible turbopack
types.

### 4.F Regression guard
Existing `memscope-replay`/`-symbols`/`-proto` unit tests stay green. The
addr2line oracle path is exercised by 4.A/4.C so it can't silently rot.

## 5. Risks & mitigations
- **Matching addr2line exactly is hard at edges** (tail calls, zero-length
  ranges, cross-unit abstract origins). Mitigation: acceptance allows reviewed
  <1% inline-count diffs; type + boundary must match exactly. Cross-unit
  `DW_AT_abstract_origin` (`DebugInfoRef`) handled explicitly (not just
  `UnitRef`).
- **`high_pc` forms** (addr vs udata offset) and **discontiguous `DW_AT_ranges`**:
  handle both when building the table and when range-testing (covered by 4.A on
  real data, which has both).
- **One-unit-at-a-time relies on sorted probes spanning units cleanly.** If a
  unit is re-touched after eviction we re-parse it (slower, still bounded). Test
  4.D with shuffled input to ensure memory stays bounded even then (time may
  rise; memory must not).
- **mmap + lifetimes** (object/gimli borrow the mmap): keep mmap owned for the
  resolver's lifetime; resolve in one scope.

## 6. Definition of done
1. 4.A/4.B/4.C green (correctness == addr2line).
2. 4.D green: memory flat across N (Δ < 256 MB from 100 → 100k).
3. 4.E: 1.8 GB recording analyzes < 4 GB and produces sane turbopack findings.
4. New symbolizer is the default; addr2line retained only as the test oracle.
</content>
