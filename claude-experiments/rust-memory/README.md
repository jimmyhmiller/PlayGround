# memscope — JVM-style memory tooling for Rust

A drop-in tracking **global allocator** that gives you what the JVM has but Rust
normally can't: a **live allocation monitor**, **type-resolved heap dumps**, and
**allocation sites** — *with full Rust type information*.

The hard part is types. `GlobalAlloc::alloc` only ever sees a `Layout` (size +
align) — it's completely type-erased. memscope recovers the concrete type by
capturing a backtrace at allocation time and joining each frame to the binary's
own **DWARF** debug info: a monomorphized `Vec::<u64>::with_capacity` has a
generic linkage name but a *concrete* `DW_TAG_template_type_parameter` (`T →
u64`). That join is what tools like heaptrack / bytehound / dhat don't do, and
it's why memscope can say `Boxed<Particle>`, `HashTable<(u64, Session)>`,
`Vec<Box<Particle>>` instead of just "1.1 MiB in 17,600 allocations".

It can also emit a **real JVM `.hprof` heap dump** — so a Rust program's heap
opens in **Eclipse MAT / VisualVM / heapster** with dominator tree, retained
sizes, and paths-to-GC-roots — and it can do this for an **unmodified binary you
never touched**, by injection (`DYLD_INSERT_LIBRARIES` / `LD_PRELOAD`). See
[Heap dumps (`.hprof`)](#heap-dumps-hprof--jvm-tooling-for-rust) and
[Zero-instrumentation](#zero-instrumentation-dump-an-unmodified-binary).

## Quick start (try it in 30 seconds)

Everything is already built in this workspace. Two terminals:

```sh
# Terminal 1 — a demo program with a churning workload + the agent attached
cargo run -p demo --release --bin serve

# Terminal 2 — attach the CLI
cargo run -p memscope-cli --release -- monitor          # live heap-by-type, refreshes each second
cargo run -p memscope-cli --release -- dump --out /tmp/heap.json   # one full type-resolved dump
cargo run -p memscope-cli --release -- graph            # reference graph: top retainers (retained size)
cargo run -p memscope-cli --release -- paths 0x<addr>   # who keeps an allocation alive
cargo run -p memscope-cli --release -- events           # raw allocation event stream
cargo run -p memscope-cli --release -- mode sampled --rate 100     # switch to low-overhead sampling
```

`graph` is the heap-analyzer view (MAT-style). It reconstructs the **object
reference graph** by reading each allocation's pointer fields (via DWARF type
layout) and computes **retained size** + the **dominator tree**:

```
nodes: 22750   edges: 16735   roots: 6015
   retained      self   out  type
    1.3 MiB    256 KiB  16600  Vec<Box<Particle>>      ← if this died, 1.3 MiB frees
  520.0 KiB    520 KiB      0  HashTable<(u64, Session)>
   55.6 KiB    1.5 KiB     50  Vec<Vec<u8>>
```

`paths <addr>` answers "who keeps this alive": the dominator chain up to the
roots plus every direct referrer (with the byte offset of the pointer).

`monitor` shows, live:

```
mode=Full  rate=1  live=1.4 MiB  total allocs=23263  total alloc'd=4.7 MiB  dropped events=0
live allocations: 16518   distinct sites: 18

   count         bytes  type (shape<element>)
────────────────────────────────────────────────────────────────
   10400     650.0 KiB  Boxed<Particle>
       1     520.0 KiB  HashTable<(u64, serve::Session)>
       1     128.0 KiB  Vec<alloc::boxed::Box<serve::Particle, ...>>
    2905      90.8 KiB  Vec<u32>
    3152      31.8 KiB  StringBuf<u8>
```

A saved dump is **self-contained** — explore it later with no live process:

```sh
cargo run -p memscope-cli --release -- show /tmp/heap.json
```

## Using it in your own program

```toml
# Cargo.toml — debug info is required for type recovery (no nightly, no toolchain change)
[profile.release]
debug = true
```

Two lines to get a live agent:

```rust
#[global_allocator]
static GLOBAL: memscope::MemScope = memscope::MemScope::system();

fn main() {
    memscope::set_mode(memscope::Mode::Full);   // or Mode::Sampled
    memscope::start_agent().unwrap();           // live: attach the CLI/UI over a socket
    // ... your program ...
}
```

…or record the whole allocation stream to a file to explore later / build a viewer:

```rust
    memscope::set_mode(memscope::Mode::Full);
    memscope::record_to_file("allocs.jsonl").unwrap();   // self-contained recording
    memscope::mark("after_warmup");                       // checkpoint to diff against later
```

Then: `memscope replay allocs.mscope` (reference reader), or parse it yourself.
Wrap a different inner allocator (jemalloc, mimalloc) with `MemScope::new(inner)`.

### Heap dumps (`.hprof`) — JVM tooling for Rust

memscope can serialize a point-in-time heap as a **JVM HPROF 1.0.2** dump. One
call, anywhere your heap is interesting:

```rust
memscope::heap_dump("heap.hprof").unwrap();   // type-resolved, self-contained
```

Open `heap.hprof` in **Eclipse MAT**, **VisualVM**, or **heapster** and you get
the full JVM heap-analysis experience — on a Rust program:

```
$ heapster heap.hprof describe 0x822c052c0
  Class:         Session
  Retained size: 592 bytes
  Fields:
    f@0x8   Object  0x823017e80  java.lang.Object[]    # the messages Vec
    f@0x18  Object  0x823001b20  User                  # the Box<User>
    f@0x20  Object  null                               # next: None
  Shortest GC root path:
    java.lang.Object[]  →  Session
```

Each Rust allocation becomes an object, each recovered type a class; struct
pointer fields (from DWARF layout) become object references, `Vec`/`HashMap`
become arrays, `String`/`Vec<u8>` become byte arrays (with real contents). The
graph's roots become HPROF GC roots, so MAT's **dominator tree, retained sizes,
paths-to-roots, and leak suspects** all work — verified end-to-end in heapster
(class histogram, retention chains, even a 30-deep linked-list dominator chain).

### Zero-instrumentation: dump an *unmodified* binary

You don't even need to change the program — or fiddle with `DYLD`/`LD_PRELOAD`
yourself. `memscope run` injects the shim, wires the dump trigger, and launches
the target for you:

```sh
memscope run --on-exit  -- ./your_program args…     # dump the final heap
memscope run --after 5s -- ./your_program args…     # dump 5s in (it keeps running)
memscope run --at-bytes 50MB -- ./your_program      # dump when the live heap first hits 50MB
memscope run --out /tmp/heap-{pid}.hprof -- ./prog  # choose the path ({pid}/{n} expand)
memscope dump-pid <pid>                             # trigger a dump in a running `run` process
```

Under the hood it's a load-time `malloc` interpose + a `SIGUSR1` trigger, exactly
like `jmap` — you can also drive it by hand:

```sh
DYLD_INSERT_LIBRARIES=target/release/libmemscope_preload.dylib ./your_program &  # macOS
LD_PRELOAD=target/release/libmemscope_preload.so ./your_program &                # Linux
kill -USR1 <pid>          # → /tmp/memscope-<pid>-0.hprof   (open in MAT/heapster)
```

**When should the snapshot be taken?** A heap dump is a point in time, and the
right point depends on intent — there's no single default:

* **`--on-exit`** — what the program *failed to free*. In Rust, RAII drops locals
  at the end of `main`, so for a build-then-exit CLI this captures almost nothing;
  it's right for *leaks* (globals, leaked `Box`, `Rc` cycles) and long-running
  servers shutting down.
* **`--at-bytes N`** — the heap *while it's big*. Robust for short programs (no
  timing race), but beware it fires the **first** time the threshold is crossed —
  which can be during *startup*, before the real working set is built.
* **`--after DUR`** / **`dump-pid`** — at **steady state**, once the program has
  done its loading/warmup. The most reliable way to capture "what is this process
  holding right now." Use `dump-pid` for daemons/servers after they've loaded.
* **Two dumps + `diff`** — for "what grew between phase A and B" (see above).

A dyld `__interpose` table routes the *target's* allocator calls through
memscope while the dylib's own allocations bind to the real `malloc` (per-image
self-exemption — so it can never recurse). Backtraces captured at `malloc` time
resolve against the **target binary's own DWARF**, so **types survive**:
injecting into a plain `Account`/`Profile` program with no memscope in it at all
still yields `Account` ×500, `Profile` ×500 in the dump. Requirements (none touch
the target's source): the default system allocator (the Rust default), and — for
type *names* — debug info / a `.dSYM` (else a complete but untyped dump). On
macOS the target must be unsigned (dev `cargo build` output is); SIP-protected /
signed binaries ignore `DYLD_INSERT_LIBRARIES`. (`DYLD_INSERT_LIBRARIES` is
inherited by child processes, so subprocesses the target spawns are dumped too.)
It captures `malloc`/`free` but **not** memory a program maps with `mmap`
directly (file-backed regions, some custom arenas) — though most Rust heap
*objects* go through `malloc`, so the bulk of a real heap is visible (heapster
loading a 197 MB dump held **gigabytes** of malloc'd buffers + materialized
objects, all captured).

The real cost shows up at **scale**: in Full mode every allocation is tracked, so
instrumenting a process with a multi-GB live heap is heavy — the tracking table
itself grows large, the target slows down, and a full dump of tens of millions of
live allocations is slow and memory-hungry. For large targets, capture **early**
(`--at-bytes`), use **`Mode::Sampled`**, or dump a representative phase rather
than the peak.

### Checkpoints & diffs (snapshot exploration)

Most memory debugging is *differential* — "it grew between request 1 and 100;
what's the delta and who retains it?" Drop a named checkpoint anywhere in your
program (costs one ring marker, no allocation tracked):

```rust
memscope::mark("after_warmup");
// … serve a request …
memscope::mark("end");
```

Then explore the recording posthoc — no live process needed:

```sh
memscope marks rec.mscope                       # list checkpoints + heap size at each
memscope diff  rec.mscope after_warmup end      # set-diff the live set between two marks
memscope diff  rec.mscope start end --json       # `start`/`end` = the stream ends; JSON for tools
```

`diff` groups the live set by `(type, site)` and reports what **grew**, what
**shrank**, and — crucially — how many of each were **born vs freed** in the
window. `born > 0, freed = 0` is the canonical "born and never died" leak
fingerprint, and each row carries the **exact source line** of the allocating
call:

```
== diff after_warmup -> end  (28 ms window) ==
   live: 202.4 KiB -> 722.1 KiB   net retained: +519.7 KiB

   GREW (live in B, not freed):
     Δcount       Δbytes    born  freed  type / site
   ───────────────────────────────────────────────────────────────────
      +5000   +263.7 KiB    5000      0  StringBuf<u8>  @ markdemo::main (markdemo.rs:42)  ← never freed
         +1   +256.0 KiB      11     10  Vec<Session>   @ markdemo::main (markdemo.rs:40)

   22 type/site groups unchanged
```

The `--json` form is built for an **AI consumer**: a token-budgeted, ranked,
source-located summary it can act on directly (then re-`diff` before/after a fix
to confirm the leak's `delta_count` went to zero). Reconstruction lives in the
reusable **`memscope-replay`** crate (`Timeline` / `LiveState`), so future tools
(an MCP wrapper) build on the same replay rather than re-parsing the stream.
Design + roadmap: `docs/ANALYSIS.md`.

### Ranked findings (`analyze`)

`diff` answers "what changed between two points"; `analyze` answers "what's
*wrong*, ranked" — over the whole recording, no marks required:

```sh
memscope analyze rec.mscope                 # ranked findings, text
memscope analyze rec.mscope --json          # same, for an AI to act on
memscope analyze rec.mscope --top 5
```

It runs a set of detectors over the event stream and emits ranked, **source-located**
findings, each with a closed-vocabulary **fix class** an agent can branch on:

```
== memory findings: rec.mscope  (2 found, showing 2) ==

[1] CHURN-STORM  severity 0.89  confidence 0.80
    Vec<u8>: 7.0 MiB allocated across 40000 allocations, 0 B live (all freed)
    @ markdemo::main (markdemo.rs:47)
    fix: reuse-buffer — hoist the allocation out of the hot loop, reuse one buffer, or use an arena.

[2] MONOTONIC-GROWTH  severity 0.45  confidence 0.85
    StringBuf<u8>: 263.7 KiB live in 5000 allocations, 0 freed (never freed)
    @ markdemo::main (markdemo.rs:42)
    fix: leak — ensure these are dropped, or bound their lifetime.
```

Detectors: **monotonic-growth** (leak / unbounded-cache — high live bytes, little
freed), **churn-storm** (huge total allocated, ~nothing live → allocating in a hot
loop), **realloc-thrash** (a buffer grown incrementally instead of `with_capacity`),
**short-lived-box** (high-volume `Box<T>` freed almost immediately → stack/pool
candidate). Findings are **merged by source location** (loop-unrolled call sites
collapse into one) and memscope's **own** profiling allocations are filtered out,
so the output is the program's issues, not the tool's. Design: `docs/ANALYSIS.md`.

### Drill into a finding (`query`)

A finding gives you a type and a site id; `query` pulls exactly the detail you
(or an agent) want next, bounded:

```sh
memscope query rec.mscope --site 169 --field stack        # full call stack, app frames marked
memscope query rec.mscope --type 'Vec<u8>' --field lifetimes  # freed-allocation lifetime histogram
memscope query rec.mscope --type 'Vec<u8>' --field sites   # every call site of this type
memscope query rec.mscope --type 'StringBuf<u8>'           # aggregate stats (default)
```

```
== lifetime histogram: Vec<u8> (9340 freed sampled) ==
        <1ms     9334  ██████████████████████████████
       1-2ms        6  █
```

### Let an AI drive it (MCP)

`memscope-mcp` is a tiny **MCP** server exposing `marks` / `diff` / `analyze` /
`query` as tools, so a Claude session can debug a heap directly — run `analyze`,
`query` the one site that matters, propose a patch, then re-`diff` to confirm the
leak's `delta_count` went to zero. Point any MCP client at the `memscope-mcp`
binary (set `MEMSCOPE_BIN` if the `memscope` CLI isn't a sibling or on `PATH`):

```jsonc
// e.g. an MCP client config
{ "command": "/path/to/memscope-mcp" }
```

Every tool returns the same token-budgeted JSON the CLI's `--json` emits.

### View it on a timeline (Perfetto)

```sh
memscope perfetto allocs.mscope --out trace.json   # convert a recording
# then open trace.json at https://ui.perfetto.dev
```

Produces a Chrome/Perfetto trace: a **`live_bytes` counter** over time plus an
**async slice for every allocation's lifetime** (alloc → free), named by
recovered type and grouped by thread — so you can scrub the timeline, see the
heap grow/shrink, and inspect when each typed object was alive. Every allocation
is emitted (no caps).

### Tag allocations with your own metadata (`meta!`)

Attach arbitrary key/value context to a scope; every allocation made inside it
inherits it (scopes nest and merge), then pivot the flame graph by any key:

```rust
let _m = memscope::meta!(subsystem = "physics");   // tag this scope
particles.push(Box::new(p));                        // → { subsystem: "physics" }
let _m = memscope::meta!(request = req.id);         // dynamic values are fine
```

```sh
memscope flamegraph rec.mscope --group-by subsystem   # subsystem value becomes the flame root
memscope flamegraph rec.mscope --filter subsystem=physics
memscope flamegraph rec.mscope --group-by phase --filter tenant=acme
```

It costs nothing per allocation — a scope records one tiny `MetaEnter`/`MetaExit`
marker on the event ring, and the reader correlates context onto allocations at
read time. Allocations outside any scope group under `<key>=<none>`. Full design
in `docs/METADATA.md`.

### Flame graph & flame chart (by call stack)

Allocations carry their full call stack, so you get both classic views:

```sh
memscope flamegraph allocs.mscope --out fg.json      # AGGREGATED by stack, width = bytes
memscope flamegraph allocs.mscope --format folded    # folded stacks (inferno / speedscope)
memscope flamegraph allocs.mscope --live             # only un-freed allocations = leak flame
memscope flamegraph allocs.mscope --no-std           # strip std/core/alloc/runtime frames
memscope flamechart allocs.mscope --out fc.json      # TIMELINE (not aggregated), x-axis = time
memscope flamechart allocs.mscope --no-std           # …same, application frames only
```

`--no-std` removes `std`/`core`/`alloc` *plumbing* (plus `hashbrown` /
`allocator_api2` — std's own HashMap + allocator shim), the lang-start + panic
machinery, the `FnOnce`/`Fn` shims, and pthread/libc entry — but **keeps the
boundary frame**: the first std call made from your code, which is what tells you
*how* it allocated (`Box::new`, `Vec::with_capacity`, `from_elem`, `collect`,
`format!`, `Vec::push`, `HashSet::insert`, `Cow::into_owned`, …). So a stack reads
`serve::main → alloc::vec::from_elem → [Vec<u8>]`. It's a *frame* filter, not a
sample one: every allocation and thread is still present, but the boilerplate is
gone, which also shrinks the output dramatically (heapster flamechart: 346 MB →
23 MB, same 836,879 allocations).

Frames carry their **source location** in the label (`heapster::scan_segment
(segment.rs:142)`) and **allocation bytes** in `args` (flamegraph: per-node total
+ width = bytes; flamechart: bytes through each frame) — so the line number and
size are visible in the viewer.

* **`flamegraph`** — merges every allocation by call stack; width = total bytes
  (or `--by count`), the recovered type is the leaf. "Where do allocations come
  from." Emitted as nested `B`/`E` events, so any Chrome-trace flame importer (or
  speedscope, via `--format folded`) renders it in true call order.
* **`flamechart`** — *not* aggregated: **every** allocation is a stack sample
  placed at its time, on **every** thread; runs of the same stack merge into one
  slice (lossless — a merged slice still represents each allocation in that run).
  Scrub the x-axis to see *what was allocating, when*. Long-lived call paths span
  wide; bursts show up as towers. Same `X`-event format. These traces are large
  (full heapster run = 346 MB / 3M slices) — that's the complete data, by design.

### The recording file format

`record_to_file` writes a **compact binary** format (`.mscope`) by default, or
newline-JSON if the path ends in `.json`/`.jsonl`. Both are **self-contained**
(resolved types + stacks embedded; a reader needs no DWARF/binary), and both are
read by `memscope replay <file>` (it auto-detects).

Binary is the one to use for real apps: a fixed **34-byte** record per event,
batch-written — ~10× smaller and much faster than JSON, written in real time.
(Measured on heapster analyzing a 576 MB heap: 1.35M events / full mode → 50 MB
binary, no stall.) Site definitions are written once each (interned), so
resolution never dominates. Layout is documented in
`crates/memscope-proto/src/recfmt.rs`.

The JSON variant (one object per line) is handy for eyeballing:

```jsonc
{"v":1,"pid":1234,"exe":"/path/to/bin"}                       // header
{"site":37,"ty":"Particle","shape":"Boxed","frames":[["serve::main","serve.rs",53,false], ...]}
{"k":"A","ts":2434000000,"a":41923141376,"sz":64,"al":8,"s":37,"t":2}   // alloc
{"k":"D","ts":2440000000,"a":41923141376,"sz":64,"al":8,"t":2}          // free
```

A `site` record is emitted once, the first time a stack is seen; events reference
it by `s` (site id). `k` = A(lloc)/D(ealloc)/R(ealloc); `a` addr, `sz` size, `al`
align, `t` thread, `ts` ns. Replay the events to reconstruct the live set at any
point — that's all the reference reader and the in-process reconstructor do.
(Recording uses Reliable mode so nothing is dropped; for very high allocation
volume use `Mode::Sampled` to keep the file small.)

## How it fits together

| crate | role |
|-------|------|
| `memscope-core` | the tracking `GlobalAlloc`: reentrancy-guarded hot path, sharded live table, site interner, event ring, runtime **Full / Sampled / Off** modes, `snapshot()` |
| `memscope-symbols` | **DWARF type recovery**: builds a linkage-name → concrete-type index + a per-type **layout index** (field offsets, pointer fields), symbolicates frames, recognizes container shapes (Box/Vec/Rc/Arc/String/HashMap) and element types |
| `memscope-graph` | **heap reference graph**: walks each allocation's pointer fields → edges → roots → dominator tree → retained sizes |
| `memscope-proto` | the wire vocabulary: hot-path `RawEvent` POD + serializable `Snapshot` / `SiteInfo` / `TypeInfo` + client/server messages |
| `memscope-agent` | in-process transport server (Unix socket, newline-JSON). Owns the type oracle and ships already-typed snapshots |
| `memscope` | thin facade: `MemScope`, mode controls, `mark()`, `start_agent()` |
| `memscope-replay` | **reusable recording reader + analysis**: parses a `.mscope`/`.jsonl` into sites/metadata/marks/events, reconstructs the live set at any checkpoint (`Timeline` / `LiveState`), classifies frames (`frames`), and runs the finding detectors (`analyze`) — the substrate for `marks` / `diff` / `analyze` |
| `memscope-cli` | terminal consumer: `monitor` / `dump` / `events` / `mode` / `show` / `marks` / `diff` / `analyze` / `query` |
| `memscope-mcp` | **MCP stdio server**: exposes `marks` / `diff` / `analyze` / `query` as tools an AI agent can call (thin wrapper over the CLI's `--json`) |
| `memscope-hprof` | **HPROF writer**: serializes a `HeapGraph` + layout as a JVM `.hprof` (instances/arrays/byte-arrays + GC roots) for Eclipse MAT / VisualVM / heapster |
| `memscope-preload` | **zero-instrumentation injector** (`cdylib`): dyld/`LD_PRELOAD` `malloc` interposition + `SIGUSR1` → `.hprof`, for dumping an unmodified binary's heap |
| `spike` | the original proof that DWARF (not demangling) is what recovers types |

## Capture modes & overhead

Measured on arm64 macOS, 1,000,000 live allocations:

| mode | time / alloc | space / live alloc |
|------|-------------|--------------------|
| Off | ~10 ns | ~0 |
| **Sampled 1/100** | **~12 ns** | ~11 B |
| **Full** | **~270 ns** | ~97 B |

* **Full** — every allocation tracked. Exact live set and heap dumps.
* **Sampled** — record ~1/N allocations; aggregates scale by N. Statistically
  accurate (the demo estimates 86,200 live from a 1/100 sample vs. 85,700 true)
  and nearly free. Switch at runtime: `memscope mode sampled --rate 100`.
* **Off** — passthrough.

The hot path uses **frame-pointer unwinding** (two memory reads per frame),
~17× cheaper than libunwind/DWARF-CFI; reliable on aarch64-apple-darwin (frame
pointers are ABI-mandated) and on x86-64 builds that keep frame pointers. Call
`set_frame_pointer_unwinding(false)` to fall back to the always-correct
`backtrace` path on builds that omit them.

## Architecture: flat hot path, off-thread reconstruction

The hot path does **no shared-state bookkeeping** — it stamps a record with a
cheap hardware timestamp and appends it to a **per-thread lock-free ring**
(`tls_ring`). No live table, no mutex, no globally-contended atomic. The live set
is rebuilt *off the allocating threads* by a single pump that drains every
thread's ring and merges them by timestamp (`sink::LiveSet`), so `snapshot()` /
the graph read reconstructed state. The consumer is **pluggable** (`EventSink`):
reconstruct in memory, stream to a file/socket/network, or fan out — see
`spawn_consumer`.

Two ring modes: **Overwrite** (wait-free; drops oldest under pressure, consumer
detects the gap) and **Reliable** (`set_ring_mode`; bounded backpressure so the
consumer doesn't lose events).

## Overhead & multi-thread scaling

Slowdown vs. the same program with no memscope (pure alloc+free churn, 12-core
machine, × = ratio to baseline):

| threads | baseline | Off | Sampled 1/100 | Full |
|---------|---------:|----:|--------------:|-----:|
| 1 | 17.9 ns | 1.1× | 1.6× | 2.8× |
| 2 |  8.8 ns | 1.1× | 2.0× | 3.5× |
| 4 |  5.5 ns | 1.1× | 1.6× | 4.5× |
| 8 |  3.5 ns | 1.0× | 1.7× | 5.2× |

Per-thread rings mean Full *scales with the cores* (absolute cost ~18 ns/op at 8
threads, down from ~50 at 1). Sampling uses a per-thread counter, so it's ~1.7×
— effectively free — across all thread counts. (The ratios look modest because
the baseline is a pure-allocation loop that parallelizes near-perfectly; real
programs allocate a fraction of the time, so end-to-end overhead is far smaller.)

## Wiring up a UI

The protocol is newline-delimited JSON over a Unix socket — language-agnostic.
A consumer sends `ClientMsg` (`GetSnapshot`, `GetStats`, `PollEvents`,
`SetMode`, …) and reads `ServerMsg` (`Snapshot`, `Stats`, `Events`, …). The
agent resolves types in-process, so the UI just renders. See
`crates/memscope-proto/src/lib.rs` for the message definitions and
`crates/memscope-cli` for a reference client.

## Platform notes

* **macOS**: DWARF lives in a `.dSYM`; memscope runs `dsymutil` automatically on
  first snapshot. Built and verified on arm64 macOS.
* **Linux**: DWARF is embedded in the ELF (with `debug = true`); read directly.
* Stable Rust. Unwinding/symbolication are abstracted behind a trait for
  portability.

## Status

Working end-to-end: tracking allocator, DWARF type recovery, runtime mode
switching, live monitor, type-resolved + posthoc heap dumps, and the **heap
reference graph** (edges, retained sizes, dominators, paths-to-roots).

The graph walks `Box`/`Rc`/`Arc`/`Vec`/`String`/structs/nested fields, **enums
discriminant-aware** (only the live variant's pointers become edges — verified
edge-exact on a mixed-variant workload), and **HashMap/HashSet** (the hashbrown
bucket layout is decoded — bucket count recovered from the allocation size,
control bytes read to find live entries). Next: allocation-age histograms,
flamegraph export, and frame-pointer unwinding for a cheaper hot path.
