# controllable-futures

A controllable async runtime (`cf-runtime`) with full visibility into task
state, an egui inspector, and a `tokio` compatibility shim (`cf-tokio`)
that runs unmodified tokio applications on top of it.

## Setup

This repo uses [`patch-crate`](https://crates.io/crates/patch-crate) to
apply a small instrumentation patch to `tokio` (see
`patches/tokio+1.43.4.patch`) instead of vendoring a full fork.

**One-time, per machine:**

```sh
cargo install patch-crate --locked
```

**Once per clone (and any time `patches/*.patch` changes):**

```sh
cargo patch-crate
```

This downloads `tokio` v1.43.4 from crates.io, applies our patch, and
writes the result to `target/patch/tokio-1.43.4/`. The workspace's
`[patch.crates-io]` redirects `tokio` to that directory.

After that, normal cargo commands work:

```sh
cargo build --workspace
cargo run -p cf-host -- demo
cargo run -p cf-turbo-demo
```

> **Heads-up:** `cargo build` cannot bootstrap `target/patch/` on its own.
> Cargo resolves `[patch.crates-io]` paths *before* any build script can
> run, so the patched directory has to exist beforehand. If you forget,
> cargo will error with `failed to read target/patch/tokio-1.43.4/Cargo.toml`
> — fix it with `cargo patch-crate`.

### Editing the tokio patch

```sh
# 1. edit files directly in target/patch/tokio-1.43.4/
# 2. regenerate the .patch file
cargo patch-crate tokio
# 3. commit patches/tokio+1.43.4.patch
```

## Layout

- `crates/` — our code
  - `cf-runtime` — the controllable executor (task metadata, Auto/Manual scheduling)
  - `cf-tokio` — tokio-shaped facade that forwards to cf-runtime
  - `cf-tokio-macros` — `#[cf_tokio::main]` proc-macro
  - `cf-host` — host binary running the egui inspector + a workload
  - `cf-ui` — the egui inspector itself
  - `cf-tracing-layer` — `tracing` layer that posts spans into cf-runtime's event log
  - `cf-trace-to-perfetto` — converts cf-runtime JSONL traces to Chrome trace format
  - `cf-tokio-hooks-test` — verifies the patched-tokio hooks fire
  - `cf-turbo-demo` — minimal demo proving turbo-tasks runs on cf-runtime
  - `cf-debug-events` — event-log debugging helper
- `patches/` — `.patch` files applied by `cargo patch-crate`
- `vendor/`
  - `mini-redis/` — used by `cf-host mini-redis` as a real test app
  - `turbo/` — slice of turbopack's `turbo-tasks-*` crates needed by `cf-turbo-demo`
