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

```rust
#[global_allocator]
static GLOBAL: memscope::MemScope = memscope::MemScope::system();

fn main() {
    memscope::set_mode(memscope::Mode::Full);   // or Mode::Sampled
    memscope::start_agent().unwrap();           // prints the socket path
    // ... your program ...
}
```

Then point the CLI (or your UI) at the socket it prints. Wrap a different inner
allocator (jemalloc, mimalloc) with `MemScope::new(inner)`.

## How it fits together

| crate | role |
|-------|------|
| `memscope-core` | the tracking `GlobalAlloc`: reentrancy-guarded hot path, sharded live table, site interner, event ring, runtime **Full / Sampled / Off** modes, `snapshot()` |
| `memscope-symbols` | **DWARF type recovery**: builds a linkage-name → concrete-type index + a per-type **layout index** (field offsets, pointer fields), symbolicates frames, recognizes container shapes (Box/Vec/Rc/Arc/String/HashMap) and element types |
| `memscope-graph` | **heap reference graph**: walks each allocation's pointer fields → edges → roots → dominator tree → retained sizes |
| `memscope-proto` | the wire vocabulary: hot-path `RawEvent` POD + serializable `Snapshot` / `SiteInfo` / `TypeInfo` + client/server messages |
| `memscope-agent` | in-process transport server (Unix socket, newline-JSON). Owns the type oracle and ships already-typed snapshots |
| `memscope` | thin facade: `MemScope`, mode controls, `start_agent()` |
| `memscope-cli` | terminal consumer: `monitor` / `dump` / `events` / `mode` / `show` |
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
